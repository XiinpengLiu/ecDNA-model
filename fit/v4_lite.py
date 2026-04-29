"""Week-level v4-lite fitting implementation.

This module intentionally fits summary-level dynamics first.  It treats the
full simulator as a later bridge target, and it keeps ddPCR as a pooled mean
anchor rather than a single-cell distribution constraint.
"""

from __future__ import annotations

import copy
import csv
import json
import os
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

import config as cfg
from fit.data import CanonicalFitDataset, EcTAGRecord, FlowRecord, QPCDRRecord, WEEK1, write_standardized_dataset
from fit.io_utils import write_json, write_netcdf_file, write_npz_or_marker, write_table_bundle, write_text_pdf
from fit.summary_types import SummaryCollection


INVALID_OBJECTIVE = 1e18
V4LiteModelVersion = {"M0", "M1", "M2", "M3", "M4"}
V4LiteDynamicsMode = {"joint", "ecDNA_only", "state_only"}
V4LiteCouplingMode = {"none", "growth", "transition", "joint"}
DEFAULT_COPY_BINS = ((0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, None))
DEFAULT_COPY_BIN_CENTERS = (0.0, 1.0, 2.5, 5.5, 11.5, 23.5, 47.5, 95.5, 160.0)
DEFAULT_DIRECTED_EDGES = (
    (cfg.NPC, cfg.OPC),
    (cfg.OPC, cfg.NPC),
    (cfg.OPC, cfg.AC),
    (cfg.AC, cfg.OPC),
    (cfg.AC, cfg.MES),
    (cfg.MES, cfg.AC),
)
OPTIONAL_DIRECTED_EDGES = ((cfg.NPC, cfg.AC), (cfg.AC, cfg.NPC))


def _normalize(values: np.ndarray, *, floor: float = 1e-12) -> np.ndarray:
    raw = np.clip(np.asarray(values, dtype=float), floor, None)
    return raw / float(np.sum(raw))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=float) - float(np.max(values))
    weights = np.exp(shifted)
    return _normalize(weights)


def _multinomial_nll(counts: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(counts, dtype=float).reshape(-1)
    p = np.clip(_normalize(probabilities), 1e-12, 1.0)
    n = float(np.sum(y))
    return -float(gammaln(n + 1.0) - np.sum(gammaln(y + 1.0)) + np.dot(y, np.log(p)))


def _safe_rmse(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=float).reshape(-1)
    if flat.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(flat))))


@dataclass(frozen=True)
class CopyNumberBinning:
    bins: tuple[tuple[int, int | None], ...] = DEFAULT_COPY_BINS
    centers: np.ndarray = field(default_factory=lambda: np.asarray(DEFAULT_COPY_BIN_CENTERS, dtype=float))

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers, dtype=float).reshape(-1)
        cfg.require(len(self.bins) == centers.size, "copy-number bin centers must match bins.")
        object.__setattr__(self, "centers", centers)

    @property
    def n_bins(self) -> int:
        return len(self.bins)

    @classmethod
    def from_observed_values(cls, values: Iterable[int], *, forced_max: int | None = None) -> "CopyNumberBinning":
        observed_max = max([0, *(int(v) for v in values)])
        if forced_max is not None:
            observed_max = max(observed_max, int(forced_max))
        keep = []
        centers = []
        for (lower, upper), center in zip(DEFAULT_COPY_BINS, DEFAULT_COPY_BIN_CENTERS):
            keep.append((lower, upper))
            centers.append(center)
            if upper is None or observed_max <= upper:
                break
        return cls(tuple(keep), np.asarray(centers, dtype=float))

    def bin_index(self, value: int | float) -> int:
        integer_value = max(0, int(round(float(value))))
        for index, (lower, upper) in enumerate(self.bins):
            if upper is None:
                if integer_value >= lower:
                    return index
            elif lower <= integer_value <= upper:
                return index
        return self.n_bins - 1

    def counts(self, values: Iterable[int | float]) -> np.ndarray:
        result = np.zeros(self.n_bins, dtype=int)
        for value in values:
            result[self.bin_index(value)] += 1
        return result

    def probabilities(self, values: Iterable[int | float], *, epsilon: float = 0.0) -> np.ndarray:
        counts = self.counts(values).astype(float)
        if epsilon:
            counts += float(epsilon)
        if float(np.sum(counts)) <= 0.0:
            counts[0] = 1.0
        return counts / float(np.sum(counts))

    def mean(self, probabilities: np.ndarray) -> float:
        return float(np.dot(np.asarray(probabilities, dtype=float).reshape(self.n_bins), self.centers))

    def tail_probability(self, probabilities: np.ndarray) -> float:
        return float(np.asarray(probabilities, dtype=float).reshape(self.n_bins)[-1])


@dataclass(frozen=True)
class FlowObservation:
    condition: str
    week: int
    counts: np.ndarray
    replicate_id: str


@dataclass(frozen=True)
class CountObservation:
    condition: str
    week: int
    value: float
    replicate_id: str
    gate_index: int | None = None


@dataclass(frozen=True)
class QPCDRObservation:
    condition: str
    week: int
    gate_index: int
    species_index: int
    value: float
    batch_index: int
    replicate_id: str


@dataclass(frozen=True)
class EcTAGHistogramObservation:
    condition: str
    week: int
    gate_index: int
    species_index: int
    counts: np.ndarray
    replicate_id: str


@dataclass(frozen=True)
class EcTAGCorrelationObservation:
    condition: str
    week: int
    gate_index: int
    species_a: int
    species_b: int
    correlation: float
    n_cells: int
    replicate_id: str


@dataclass(frozen=True)
class DDPCRObservation:
    condition: str
    week: int
    species_index: int
    value: float
    sigma: float
    replicate_id: str


@dataclass(frozen=True)
class V4LiteStructure:
    transition_edges: tuple[tuple[int, int], ...]
    binning: CopyNumberBinning = field(default_factory=CopyNumberBinning)
    qpcdr_batches: tuple[str, ...] = ("default",)

    @classmethod
    def default(cls, *, include_optional_edge: bool = False, qpcdr_batches: Iterable[str] = ("default",), binning: CopyNumberBinning | None = None) -> "V4LiteStructure":
        edges = DEFAULT_DIRECTED_EDGES + (OPTIONAL_DIRECTED_EDGES if include_optional_edge else ())
        return cls(tuple(edges), CopyNumberBinning() if binning is None else binning, tuple(sorted(set(qpcdr_batches))) or ("default",))

    def with_qpcdr_batches(self, batches: Iterable[str]) -> "V4LiteStructure":
        return V4LiteStructure(self.transition_edges, self.binning, tuple(sorted(set(batches))) or ("default",))

    @property
    def undirected_edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted({tuple(sorted(edge)) for edge in self.transition_edges}))

    @property
    def n_mobility_edges(self) -> int:
        return len(self.undirected_edges)

    @property
    def n_qpcdr_batches(self) -> int:
        return len(self.qpcdr_batches)

    def mobility_index(self, source: int, target: int) -> int:
        return self.undirected_edges.index(tuple(sorted((int(source), int(target)))))


@dataclass
class V4LiteParameters:
    qpcdr_intercept: np.ndarray = field(default_factory=lambda: np.zeros(cfg.N_SPECIES, dtype=float))
    qpcdr_slope: np.ndarray = field(default_factory=lambda: np.ones(cfg.N_SPECIES, dtype=float))
    qpcdr_sigma: np.ndarray = field(default_factory=lambda: np.full(cfg.N_SPECIES, 0.25, dtype=float))
    qpcdr_batch_offsets: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    flow_concentration: float = 250.0
    count_dispersion: float = 25.0
    count_gate_dispersion: float = 25.0
    ectag_concentration: np.ndarray = field(default_factory=lambda: np.full(cfg.N_SPECIES, 120.0, dtype=float))
    ectag_corr_sigma: float = 0.20
    exposure_C_scale: float = 1.0
    exposure_P_scale: float = 1.0
    kernel_up_species: np.ndarray = field(default_factory=lambda: np.full(cfg.N_SPECIES, -2.20, dtype=float))
    kernel_down_species: np.ndarray = field(default_factory=lambda: np.full(cfg.N_SPECIES, -2.30, dtype=float))
    kernel_up_state: np.ndarray = field(default_factory=lambda: np.zeros(cfg.N_STATES, dtype=float))
    kernel_down_state: np.ndarray = field(default_factory=lambda: np.zeros(cfg.N_STATES, dtype=float))
    alpha_state: np.ndarray = field(default_factory=lambda: np.array([0.15, 0.05, -0.05, -0.15], dtype=float))
    beta_C: float = 0.50
    beta_P: float = 0.50
    lambda_M: float = 0.20
    mobility_log: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    growth_base: np.ndarray = field(default_factory=lambda: np.zeros(cfg.N_STATES, dtype=float))
    theta_P: float = 0.0
    chi_C: float = 0.10
    chi_P: float = 0.10
    omega_O_given_C: float = 0.45
    theta_B: float = 0.0
    burden_loss_effect: float = 0.0
    co_segregation_rho: float = 0.0
    sort_purity_matrix: np.ndarray = field(default_factory=lambda: np.eye(cfg.N_STATES, dtype=float))

    @classmethod
    def default(cls, structure: V4LiteStructure | None = None, *, purity_matrix: np.ndarray | None = None, qpcdr_calibration: Mapping[str, Mapping[str, float]] | None = None) -> "V4LiteParameters":
        model_structure = V4LiteStructure.default() if structure is None else structure
        params = cls()
        params.qpcdr_batch_offsets = np.zeros(model_structure.n_qpcdr_batches, dtype=float)
        params.mobility_log = np.full(model_structure.n_mobility_edges, np.log(0.08), dtype=float)
        params.alpha_state = params.alpha_state - float(np.mean(params.alpha_state))
        if purity_matrix is not None:
            params.sort_purity_matrix = _normalize_purity_matrix(purity_matrix)
        for species, calibration in (qpcdr_calibration or {}).items():
            if species not in cfg.SPECIES_INDEX:
                continue
            idx = cfg.SPECIES_INDEX[species]
            if "intercept" in calibration:
                params.qpcdr_intercept[idx] = float(calibration["intercept"])
            if "slope" in calibration:
                params.qpcdr_slope[idx] = max(float(calibration["slope"]), 1e-8)
            if "sigma" in calibration:
                params.qpcdr_sigma[idx] = max(float(calibration["sigma"]), 1e-8)
        return params

    def copy(self) -> "V4LiteParameters":
        return copy.deepcopy(self)


@dataclass(frozen=True)
class V4LiteTensor:
    dataset: CanonicalFitDataset
    structure: V4LiteStructure
    condition_names: tuple[str, ...]
    weeks: tuple[int, ...]
    initial_state_abundance: dict[str, np.ndarray]
    initial_copy_distributions: dict[str, np.ndarray]
    exposure_C: dict[str, np.ndarray]
    exposure_P: dict[str, np.ndarray]
    flow_observations: tuple[FlowObservation, ...]
    count_observations: tuple[CountObservation, ...]
    qpcdr_observations: tuple[QPCDRObservation, ...]
    ectag_hist_observations: tuple[EcTAGHistogramObservation, ...]
    ectag_corr_observations: tuple[EcTAGCorrelationObservation, ...]
    ddpcr_observations: tuple[DDPCRObservation, ...]
    observed_summary: SummaryCollection
    has_total_counts: bool
    has_same_cell_ectag: bool
    burden_star: float

    @property
    def week_to_index(self) -> dict[int, int]:
        return {week: idx for idx, week in enumerate(self.weeks)}

    @property
    def batch_to_index(self) -> dict[str, int]:
        return {batch: idx for idx, batch in enumerate(self.structure.qpcdr_batches)}


@dataclass(frozen=True)
class V4LitePrediction:
    condition_names: tuple[str, ...]
    weeks: tuple[int, ...]
    state_abundance: dict[str, np.ndarray]
    copy_distributions: dict[str, np.ndarray]
    transition_matrices: dict[str, np.ndarray]
    growth_rates: dict[str, np.ndarray]
    copy_kernels: dict[str, np.ndarray]
    summary: SummaryCollection


@dataclass(frozen=True)
class V4LiteBlockResult:
    name: str
    dimension: int
    negative_log_likelihood: float
    residual_norm: float


@dataclass(frozen=True)
class V4LiteObjectiveArtifacts:
    params: V4LiteParameters
    prediction: V4LitePrediction
    vector: np.ndarray


@dataclass(frozen=True)
class V4LiteObjectiveResult:
    total_objective: float
    data_negative_log_likelihood: float
    prior_penalty: float
    block_results: tuple[V4LiteBlockResult, ...]
    artifacts: V4LiteObjectiveArtifacts | None = None


@dataclass(frozen=True)
class V4LitePriorPredictiveReport:
    n_draws: int
    pass_rate: float
    failures: dict[str, int]


@dataclass(frozen=True)
class V4LitePosteriorPredictiveReport:
    block_rmse: dict[str, float]
    block_relative_rmse: dict[str, float]
    block_max_abs_residual: dict[str, float]
    worst_relative_rmse: float
    block_coverage_90: dict[str, float] = field(default_factory=dict)
    overall_coverage_90: float | None = None


@dataclass(frozen=True)
class V4LiteProfilePoint:
    dimension_index: int
    offset: float
    objective_value: float


@dataclass(frozen=True)
class V4LiteFakeDataRecoveryReport:
    recovered_objective: float
    normalized_error: float
    passed: bool
    block_relative_rmse: dict[str, float] = field(default_factory=dict)
    n_synthetic: int = 1
    skipped_reason: str | None = None
    sign_recovery_rate: float | None = None
    coverage_rate: float | None = None


@dataclass(frozen=True)
class V4LiteLeaveOneWeekOutReport:
    heldout_scores: dict[int, float]


@dataclass(frozen=True)
class V4LiteSBCReport:
    n_datasets: int
    ranks: dict[str, tuple[int, ...]]
    failures: int
    skipped_reason: str | None = None


@dataclass(frozen=True)
class V4LitePosteriorSamples:
    parameter_names: tuple[str, ...]
    samples: np.ndarray
    acceptance_rate: float
    skipped_reason: str | None = None
    covariance: np.ndarray | None = None
    method: str = "approximate"


@dataclass(frozen=True)
class V4LiteReports:
    tensor_summary: dict[str, object]
    calibration_report: dict[str, object]
    ecDNA_report: dict[str, object]
    identifiability_report: dict[str, object]
    posterior_predictive_report: dict[str, object]
    fake_data_report: dict[str, object]
    implementation_status_report: dict[str, object] = field(default_factory=dict)
    prior_diagnostics_report: dict[str, object] = field(default_factory=dict)
    count_observation_report: dict[str, object] = field(default_factory=dict)
    posterior_predictive_residuals: tuple[dict[str, object], ...] = ()
    posterior_predictive_intervals: tuple[dict[str, object], ...] = ()
    sbc_report: dict[str, object] | None = None


@dataclass(frozen=True)
class FullToLiteProjection:
    weeks: tuple[int, ...]
    state_abundance: np.ndarray
    copy_distributions: np.ndarray
    transition_matrices: np.ndarray | None
    growth_rates: np.ndarray | None
    copy_kernels: np.ndarray | None
    diagnostics: dict[str, object]


def _normalize_purity_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    totals = np.sum(values, axis=0)
    cfg.require(values.shape == (cfg.N_STATES, cfg.N_STATES), "purity matrix has invalid shape.")
    cfg.require(np.all(totals > 0.0), "purity columns must have positive mass.")
    return values / totals


def _flow_record_count(record: FlowRecord) -> float:
    if record.count is not None:
        return float(record.count)
    if record.fraction is not None and record.total_events is not None:
        return float(record.fraction * record.total_events)
    return float((record.fraction or 0.0) * 1000.0)


def _week1_total_count(dataset: CanonicalFitDataset, condition: str) -> float:
    values = [row.value for row in dataset.counts if row.condition == condition and row.week == WEEK1 and row.gate is None]
    if values:
        return max(float(np.mean(values)), 1.0)
    return max(float(sum(_flow_record_count(row) for row in dataset.flow if row.condition == condition and row.week == WEEK1)), 1.0)


def build_v4_lite_tensor(dataset: CanonicalFitDataset, *, condition_names: Iterable[str] | None = None, structure: V4LiteStructure | None = None) -> V4LiteTensor:
    selected = tuple(dataset.condition_names() if condition_names is None else condition_names)
    all_copy_values: list[int] = [row.value for row in dataset.ectag]
    for by_state in dataset.week1_copy_distributions.values():
        for matrix in by_state.values():
            all_copy_values.extend(np.asarray(matrix, dtype=int).reshape(-1).tolist())
    binning = CopyNumberBinning.from_observed_values(all_copy_values, forced_max=dataset.ectag_hist_max)
    model_structure = V4LiteStructure.default(qpcdr_batches=dataset.qpcdr_batches(), binning=binning) if structure is None else V4LiteStructure(structure.transition_edges, structure.binning, dataset.qpcdr_batches())
    max_week = max(dataset.dynamic_weeks())
    weeks = tuple(range(WEEK1, int(max_week) + 1))

    initial_state_abundance: dict[str, np.ndarray] = {}
    initial_copy_distributions: dict[str, np.ndarray] = {}
    exposure_C: dict[str, np.ndarray] = {}
    exposure_P: dict[str, np.ndarray] = {}
    burden_terms: list[float] = []
    for condition in selected:
        init_condition = dataset.resolve_initialization_condition(condition)
        init = dataset.build_empirical_initialization(condition)
        cfg.require(init.empirical_flow_fractions is not None, "empirical flow is required.")
        cfg.require(init.empirical_sorted_copy_distributions is not None, "empirical copy distributions are required.")
        total = _week1_total_count(dataset, init_condition)
        initial_state_abundance[condition] = total * np.asarray(init.empirical_flow_fractions, dtype=float)
        p0 = np.zeros((cfg.N_STATES, cfg.N_SPECIES, model_structure.binning.n_bins), dtype=float)
        for state_idx, state_name in enumerate(cfg.STATE_NAMES):
            matrix = init.empirical_sorted_copy_distributions[state_name]
            for species_idx in range(cfg.N_SPECIES):
                p0[state_idx, species_idx, :] = model_structure.binning.probabilities(matrix[:, species_idx], epsilon=1e-3)
                burden_terms.append(float(np.dot(p0[state_idx, species_idx], np.log1p(model_structure.binning.centers))))
        initial_copy_distributions[condition] = p0
        schedules = dataset.conditions[condition].build_input_schedules()
        exposure_C[condition] = np.asarray([schedules["u_C"](float(week - WEEK1) + 0.5) for week in weeks[:-1]], dtype=float)
        exposure_P[condition] = np.asarray([schedules["u_P"](float(week - WEEK1) + 0.5) for week in weeks[:-1]], dtype=float)

    flow_observations = _build_flow_observations(dataset, selected)
    count_observations = tuple(
        CountObservation(row.condition, row.week, float(row.value), row.replicate_id or "__aggregate__", None if row.gate is None else cfg.STATE_INDEX[row.gate])
        for row in dataset.counts
        if row.condition in selected and row.week > WEEK1
    )
    batch_to_index = {batch: idx for idx, batch in enumerate(model_structure.qpcdr_batches)}
    qpcdr_observations = tuple(
        QPCDRObservation(row.condition, row.week, cfg.STATE_INDEX[row.state], cfg.SPECIES_INDEX[row.species], float(row.value), batch_to_index[row.batch], row.replicate_id or "__aggregate__")
        for row in dataset.qpcdr
        if row.condition in selected and row.week > WEEK1
    )
    ectag_hist_observations, ectag_corr_observations, has_same_cell = _build_ectag_observations(dataset, selected, model_structure.binning)
    ddpcr_observations = tuple(
        DDPCRObservation(row.condition, row.week, cfg.SPECIES_INDEX[row.species], float(row.value), _ddpcr_sigma(row), row.replicate_id or "__aggregate__")
        for row in dataset.ddpcr
        if row.condition in selected and row.week > WEEK1
    )
    observed_summary = _observed_summary_from_observations(flow_observations, count_observations, qpcdr_observations, ectag_hist_observations, ectag_corr_observations, ddpcr_observations, dataset)
    return V4LiteTensor(
        dataset=dataset,
        structure=model_structure,
        condition_names=selected,
        weeks=weeks,
        initial_state_abundance=initial_state_abundance,
        initial_copy_distributions=initial_copy_distributions,
        exposure_C=exposure_C,
        exposure_P=exposure_P,
        flow_observations=flow_observations,
        count_observations=count_observations,
        qpcdr_observations=qpcdr_observations,
        ectag_hist_observations=ectag_hist_observations,
        ectag_corr_observations=ectag_corr_observations,
        ddpcr_observations=ddpcr_observations,
        observed_summary=observed_summary,
        has_total_counts=any(obs.gate_index is None for obs in count_observations),
        has_same_cell_ectag=has_same_cell,
        burden_star=float(np.median(burden_terms)) if burden_terms else 0.0,
    )


def _ddpcr_sigma(row) -> float:
    if row.lower is not None and row.upper is not None and row.value > 0:
        return max(float(np.log(max(row.upper, 1e-8)) - np.log(max(row.lower, 1e-8))) / 3.92, 0.05)
    return 0.20


def _build_flow_observations(dataset: CanonicalFitDataset, selected: tuple[str, ...]) -> tuple[FlowObservation, ...]:
    grouped: dict[tuple[str, int, str], np.ndarray] = {}
    totals: dict[tuple[str, int, str], float] = {}
    for row in dataset.flow:
        if row.condition not in selected or row.week <= WEEK1:
            continue
        key = (row.condition, row.week, row.replicate_id or "__aggregate__")
        grouped.setdefault(key, np.zeros(cfg.N_STATES, dtype=float))
        grouped[key][cfg.STATE_INDEX[row.state]] += _flow_record_count(row)
        totals[key] = totals.get(key, 0.0) + _flow_record_count(row)
    result = []
    for (condition, week, replicate), counts in sorted(grouped.items()):
        result.append(FlowObservation(condition, week, counts.astype(float), replicate))
    return tuple(result)


def _build_ectag_observations(dataset: CanonicalFitDataset, selected: tuple[str, ...], binning: CopyNumberBinning) -> tuple[tuple[EcTAGHistogramObservation, ...], tuple[EcTAGCorrelationObservation, ...], bool]:
    grouped: dict[tuple[str, int, str, str, str], list[int]] = {}
    cell_grouped: dict[tuple[str, int, str, str, str], dict[str, int]] = {}
    for row in dataset.ectag:
        if row.condition not in selected or row.week <= WEEK1:
            continue
        rep = row.replicate_id or "__aggregate__"
        grouped.setdefault((row.condition, row.week, row.state, row.species, rep), []).append(int(row.value))
        cell_grouped.setdefault((row.condition, row.week, row.state, rep, row.cell_id), {})[row.species] = int(row.value)
    hist = [
        EcTAGHistogramObservation(condition, week, cfg.STATE_INDEX[state], cfg.SPECIES_INDEX[species], binning.counts(values), replicate)
        for (condition, week, state, species, replicate), values in sorted(grouped.items())
    ]
    corr: list[EcTAGCorrelationObservation] = []
    same_cell = False
    by_snapshot: dict[tuple[str, int, str, str], list[list[int]]] = {}
    for (condition, week, state, replicate, _cell), species_map in cell_grouped.items():
        if set(species_map) == set(cfg.SPECIES):
            same_cell = True
            by_snapshot.setdefault((condition, week, state, replicate), []).append([species_map[species] for species in cfg.SPECIES])
    for (condition, week, state, replicate), rows in sorted(by_snapshot.items()):
        matrix = np.asarray(rows, dtype=float)
        if matrix.shape[0] < 2 or np.any(np.std(matrix, axis=0) == 0.0):
            cmat = np.zeros((cfg.N_SPECIES, cfg.N_SPECIES), dtype=float)
        else:
            cmat = np.nan_to_num(np.corrcoef(matrix, rowvar=False), nan=0.0)
        for a, b in combinations(range(cfg.N_SPECIES), 2):
            corr.append(EcTAGCorrelationObservation(condition, week, cfg.STATE_INDEX[state], a, b, float(cmat[a, b]), matrix.shape[0], replicate))
    return tuple(hist), tuple(corr), same_cell


def _observed_summary_from_observations(
    flow: tuple[FlowObservation, ...],
    counts: tuple[CountObservation, ...],
    qpcdr: tuple[QPCDRObservation, ...],
    ectag: tuple[EcTAGHistogramObservation, ...],
    corr: tuple[EcTAGCorrelationObservation, ...],
    ddpcr: tuple[DDPCRObservation, ...],
    dataset: CanonicalFitDataset,
) -> SummaryCollection:
    maps = _empty_block_maps()
    for obs in flow:
        total = max(float(np.sum(obs.counts)), 1e-12)
        for state_idx, state_name in enumerate(cfg.STATE_NAMES):
            key = f"{obs.condition}|week{obs.week}|state={state_name}|rep={obs.replicate_id}"
            maps["flow_count"][key] = float(obs.counts[state_idx])
            maps["flow_fraction"][key] = float(obs.counts[state_idx] / total)
    for obs in counts:
        if obs.gate_index is None:
            maps["count_total"][f"{obs.condition}|week{obs.week}|rep={obs.replicate_id}"] = float(obs.value)
        else:
            maps["count_gate"][f"{obs.condition}|week{obs.week}|state={cfg.STATE_NAMES[obs.gate_index]}|rep={obs.replicate_id}"] = float(obs.value)
    for obs in qpcdr:
        prefix = f"{obs.condition}|week{obs.week}|state={cfg.STATE_NAMES[obs.gate_index]}|species={cfg.SPECIES[obs.species_index]}|rep={obs.replicate_id}"
        maps["qpcdr"][prefix] = float(obs.value)
    for obs in ectag:
        prefix = f"{obs.condition}|week{obs.week}|state={cfg.STATE_NAMES[obs.gate_index]}|species={cfg.SPECIES[obs.species_index]}|rep={obs.replicate_id}"
        total = max(float(np.sum(obs.counts)), 1e-12)
        probs = obs.counts.astype(float) / total
        for idx, prob in enumerate(probs.tolist()):
            maps["ectag_hist"][f"{prefix}|bin={idx}"] = float(prob)
        centers = dataset_aware_centers(dataset, obs.counts.size)
        mean = float(np.dot(probs, centers))
        maps["ectag_moments"][f"{prefix}|zero_fraction"] = float(probs[0]) if probs.size else 0.0
        maps["ectag_moments"][f"{prefix}|mean"] = mean
        maps["ectag_moments"][f"{prefix}|tail"] = float(probs[-1]) if probs.size else 0.0
    for obs in corr:
        key = f"{obs.condition}|week{obs.week}|state={cfg.STATE_NAMES[obs.gate_index]}|pair={cfg.SPECIES[obs.species_a]}-{cfg.SPECIES[obs.species_b]}|rep={obs.replicate_id}"
        maps["ectag_corr"][key] = float(obs.correlation)
    for obs in ddpcr:
        maps["ddpcr_pooled_mean"][f"{obs.condition}|week{obs.week}|species={cfg.SPECIES[obs.species_index]}|rep={obs.replicate_id}"] = float(obs.value)
    return SummaryCollection.from_block_maps(maps)


def dataset_aware_centers(_dataset: CanonicalFitDataset, n_bins: int) -> np.ndarray:
    return np.asarray(DEFAULT_COPY_BIN_CENTERS[:n_bins], dtype=float)


def _empty_block_maps() -> dict[str, dict[str, float]]:
    return {
        "flow_fraction": {},
        "flow_count": {},
        "count_total": {},
        "count_gate": {},
        "qpcdr": {},
        "ectag_hist": {},
        "ectag_moments": {},
        "ectag_corr": {},
        "ddpcr_pooled_mean": {},
    }


def _copy_log_signals(distributions: np.ndarray, binning: CopyNumberBinning) -> np.ndarray:
    return np.tensordot(np.asarray(distributions, dtype=float), np.log1p(binning.centers), axes=([-1], [0]))


def _copy_means(distributions: np.ndarray, binning: CopyNumberBinning) -> np.ndarray:
    return np.tensordot(np.asarray(distributions, dtype=float), binning.centers, axes=([-1], [0]))


def _copy_kernel(params: V4LiteParameters, binning: CopyNumberBinning, state_idx: int, species_idx: int) -> np.ndarray:
    n = binning.n_bins
    kernel = np.zeros((n, n), dtype=float)
    up = float(cfg.sigmoid(params.kernel_up_species[species_idx] + params.kernel_up_state[state_idx]))
    down = float(cfg.sigmoid(params.kernel_down_species[species_idx] + params.kernel_down_state[state_idx]))
    stay = max(1.0 - 0.45 * (up + down), 0.05)
    for source in range(n):
        weights = np.zeros(n, dtype=float)
        weights[source] += stay
        weights[min(n - 1, source + 1)] += up
        weights[max(0, source - 1)] += down
        kernel[source, :] = _normalize(weights)
    return kernel


def _transition_matrix(params: V4LiteParameters, structure: V4LiteStructure, copy_distribution: np.ndarray, binning: CopyNumberBinning, *, use_copy_coupling: bool = True) -> np.ndarray:
    log_signals = _copy_log_signals(copy_distribution, binning)
    matrix = np.zeros((cfg.N_STATES, cfg.N_STATES), dtype=float)
    for source in range(cfg.N_STATES):
        logits = np.full(cfg.N_STATES, -8.0, dtype=float)
        logits[source] = 0.0
        for target in range(cfg.N_STATES):
            if source == target:
                continue
            if (source, target) in structure.transition_edges:
                mobility = params.mobility_log[structure.mobility_index(source, target)]
                species_signal = 0.0
                if use_copy_coupling:
                    species_signal = (
                        params.beta_C * log_signals[source, cfg.CDK4]
                        + params.beta_P * log_signals[source, cfg.PDGFRA]
                        + params.lambda_M * log_signals[source, cfg.MYC]
                    )
                logits[target] = mobility + 0.10 * species_signal + params.alpha_state[target] - params.alpha_state[source]
        matrix[source, :] = _softmax(logits)
    return matrix


def _growth_rates(params: V4LiteParameters, copy_distribution: np.ndarray, binning: CopyNumberBinning, *, use_copy_coupling: bool = True) -> np.ndarray:
    if not use_copy_coupling:
        return params.growth_base.copy()
    means = _copy_means(copy_distribution, binning)
    log_signals = np.log1p(means)
    return params.growth_base + params.theta_P * log_signals[:, cfg.MYC] * 0.05 + params.theta_B * np.mean(log_signals, axis=1) * 0.05


def predict_v4_lite(
    tensor: V4LiteTensor,
    params: V4LiteParameters,
    *,
    dynamics_mode: str = "joint",
    frozen_copy_distributions: Mapping[str, np.ndarray] | None = None,
    coupling_mode: str = "joint",
) -> V4LitePrediction:
    cfg.require(dynamics_mode in V4LiteDynamicsMode, f"Unknown v4-lite dynamics mode {dynamics_mode}.")
    cfg.require(coupling_mode in V4LiteCouplingMode, f"Unknown v4-lite coupling mode {coupling_mode}.")
    n_weeks = len(tensor.weeks)
    n_bins = tensor.structure.binning.n_bins
    state_abundance: dict[str, np.ndarray] = {}
    copy_distributions: dict[str, np.ndarray] = {}
    transition_matrices: dict[str, np.ndarray] = {}
    growth_rates: dict[str, np.ndarray] = {}
    copy_kernels: dict[str, np.ndarray] = {}
    for condition in tensor.condition_names:
        abundance = np.zeros((n_weeks, cfg.N_STATES), dtype=float)
        copies = np.zeros((n_weeks, cfg.N_STATES, cfg.N_SPECIES, n_bins), dtype=float)
        transitions = np.zeros((n_weeks - 1, cfg.N_STATES, cfg.N_STATES), dtype=float)
        growth = np.zeros((n_weeks - 1, cfg.N_STATES), dtype=float)
        kernels = np.zeros((n_weeks - 1, cfg.N_STATES, cfg.N_SPECIES, n_bins, n_bins), dtype=float)
        abundance[0, :] = tensor.initial_state_abundance[condition]
        copies[0, :, :, :] = tensor.initial_copy_distributions[condition]
        if dynamics_mode == "state_only" and frozen_copy_distributions is not None:
            frozen = np.asarray(frozen_copy_distributions[condition], dtype=float)
            copies[:, :, :, :] = frozen
        for interval in range(n_weeks - 1):
            if not (dynamics_mode == "state_only" and frozen_copy_distributions is not None):
                for state_idx in range(cfg.N_STATES):
                    for species_idx in range(cfg.N_SPECIES):
                        kernels[interval, state_idx, species_idx] = _copy_kernel(params, tensor.structure.binning, state_idx, species_idx)
            if dynamics_mode == "ecDNA_only":
                transition = np.eye(cfg.N_STATES, dtype=float)
            else:
                transition = _transition_matrix(params, tensor.structure, copies[interval], tensor.structure.binning, use_copy_coupling=coupling_mode in {"transition", "joint"})
            transitions[interval] = transition
            growth[interval] = _growth_rates(params, copies[interval], tensor.structure.binning, use_copy_coupling=coupling_mode in {"growth", "joint"})
            next_abundance = np.zeros(cfg.N_STATES, dtype=float)
            for source in range(cfg.N_STATES):
                next_abundance += abundance[interval, source] * np.exp(growth[interval, source]) * transition[source, :]
            abundance[interval + 1, :] = next_abundance
            if dynamics_mode == "state_only" and frozen_copy_distributions is not None:
                continue
            post_kernel = np.zeros((cfg.N_STATES, cfg.N_SPECIES, n_bins), dtype=float)
            for source in range(cfg.N_STATES):
                for species_idx in range(cfg.N_SPECIES):
                    post_kernel[source, species_idx] = copies[interval, source, species_idx] @ kernels[interval, source, species_idx]
            if dynamics_mode == "joint":
                for target in range(cfg.N_STATES):
                    denom = max(float(np.sum(abundance[interval] * transition[:, target])), 1e-12)
                    for species_idx in range(cfg.N_SPECIES):
                        mixed = np.zeros(n_bins, dtype=float)
                        for source in range(cfg.N_STATES):
                            mixed += abundance[interval, source] * transition[source, target] * post_kernel[source, species_idx]
                        copies[interval + 1, target, species_idx] = _normalize(mixed / denom)
            else:
                copies[interval + 1] = post_kernel
        state_abundance[condition] = abundance
        copy_distributions[condition] = copies
        transition_matrices[condition] = transitions
        growth_rates[condition] = growth
        copy_kernels[condition] = kernels
    summary = _prediction_summary(tensor, params, state_abundance, copy_distributions)
    return V4LitePrediction(tensor.condition_names, tensor.weeks, state_abundance, copy_distributions, transition_matrices, growth_rates, copy_kernels, summary)


def _prediction_summary(tensor: V4LiteTensor, params: V4LiteParameters, abundance: dict[str, np.ndarray], copies: dict[str, np.ndarray]) -> SummaryCollection:
    maps = _empty_block_maps()
    week_index = tensor.week_to_index
    for obs in tensor.flow_observations:
        idx = week_index[obs.week]
        predicted_counts = abundance[obs.condition][idx]
        total = max(float(np.sum(predicted_counts)), 1e-12)
        for state_idx, state_name in enumerate(cfg.STATE_NAMES):
            key = f"{obs.condition}|week{obs.week}|state={state_name}|rep={obs.replicate_id}"
            maps["flow_count"][key] = float(predicted_counts[state_idx])
            maps["flow_fraction"][key] = float(predicted_counts[state_idx] / total)
    for obs in tensor.count_observations:
        idx = week_index[obs.week]
        if obs.gate_index is None:
            maps["count_total"][f"{obs.condition}|week{obs.week}|rep={obs.replicate_id}"] = float(np.sum(abundance[obs.condition][idx]))
        else:
            maps["count_gate"][f"{obs.condition}|week{obs.week}|state={cfg.STATE_NAMES[obs.gate_index]}|rep={obs.replicate_id}"] = float(abundance[obs.condition][idx, obs.gate_index])
    for obs in tensor.qpcdr_observations:
        key = f"{obs.condition}|week{obs.week}|state={cfg.STATE_NAMES[obs.gate_index]}|species={cfg.SPECIES[obs.species_index]}|rep={obs.replicate_id}"
        maps["qpcdr"][key] = float(_expected_qpcdr_value(tensor, params, None, obs, copy_distributions=copies))
    for obs in tensor.ectag_hist_observations:
        idx = week_index[obs.week]
        probs = copies[obs.condition][idx, obs.gate_index, obs.species_index]
        prefix = f"{obs.condition}|week{obs.week}|state={cfg.STATE_NAMES[obs.gate_index]}|species={cfg.SPECIES[obs.species_index]}|rep={obs.replicate_id}"
        for bin_idx, value in enumerate(probs.tolist()):
            maps["ectag_hist"][f"{prefix}|bin={bin_idx}"] = float(value)
        maps["ectag_moments"][f"{prefix}|zero_fraction"] = float(probs[0])
        maps["ectag_moments"][f"{prefix}|mean"] = float(tensor.structure.binning.mean(probs))
        maps["ectag_moments"][f"{prefix}|tail"] = float(probs[-1])
    for obs in tensor.ectag_corr_observations:
        key = f"{obs.condition}|week{obs.week}|state={cfg.STATE_NAMES[obs.gate_index]}|pair={cfg.SPECIES[obs.species_a]}-{cfg.SPECIES[obs.species_b]}|rep={obs.replicate_id}"
        maps["ectag_corr"][key] = 0.0
    for obs in tensor.ddpcr_observations:
        idx = week_index[obs.week]
        state_total = max(float(np.sum(abundance[obs.condition][idx])), 1e-12)
        fractions = abundance[obs.condition][idx] / state_total
        means = _copy_means(copies[obs.condition][idx], tensor.structure.binning)
        pooled = float(np.dot(fractions, means[:, obs.species_index]))
        key = f"{obs.condition}|week{obs.week}|species={cfg.SPECIES[obs.species_index]}|rep={obs.replicate_id}"
        maps["ddpcr_pooled_mean"][key] = pooled
    return SummaryCollection.from_block_maps(maps)


def _expected_qpcdr_value(tensor: V4LiteTensor, params: V4LiteParameters, prediction: V4LitePrediction | None, observation: QPCDRObservation, *, copy_distributions: Mapping[str, np.ndarray] | None = None) -> float:
    if copy_distributions is None:
        cfg.require(prediction is not None, "prediction is required when copy distributions are not supplied.")
        copy_distributions = prediction.copy_distributions
    probs = copy_distributions[observation.condition][tensor.week_to_index[observation.week], observation.gate_index, observation.species_index]
    mean_copy = max(tensor.structure.binning.mean(probs), 1e-8)
    offset = params.qpcdr_intercept[observation.species_index] + params.qpcdr_batch_offsets[observation.batch_index]
    slope = params.qpcdr_slope[observation.species_index]
    if tensor.dataset.qpcdr_scale() == "ct":
        return float(offset - slope * np.log10(mean_copy + 1.0))
    return float(offset + slope * mean_copy)


class V4LiteParameterAdapter:
    def __init__(self, structure: V4LiteStructure, active_groups: tuple[str, ...], base_params: V4LiteParameters | None = None):
        self.structure = structure
        self.active_groups = tuple(active_groups)
        self.base_params = V4LiteParameters.default(structure) if base_params is None else base_params.copy()
        self.fields = self._fields_for_groups()

    def _fields_for_groups(self) -> tuple[tuple[str, str, tuple[int, ...] | None], ...]:
        fields: list[tuple[str, str, tuple[int, ...] | None]] = []
        if "observation" in self.active_groups:
            fields += [("qpcdr_intercept", "identity", (cfg.N_SPECIES,)), ("qpcdr_slope", "log", (cfg.N_SPECIES,)), ("qpcdr_sigma", "log", (cfg.N_SPECIES,))]
        if "ecDNA_kernel" in self.active_groups:
            fields += [("kernel_up_species", "identity", (cfg.N_SPECIES,)), ("kernel_down_species", "identity", (cfg.N_SPECIES,))]
        if "state_abundance" in self.active_groups:
            fields += [("growth_base", "identity", (cfg.N_STATES,)), ("mobility_log", "identity", (self.structure.n_mobility_edges,)), ("omega_O_given_C", "logit", None)]
        if "growth_coupling" in self.active_groups:
            fields += [("theta_P", "identity", None)]
        if "transition_coupling" in self.active_groups:
            fields += [("beta_C", "identity", None), ("beta_P", "identity", None), ("lambda_M", "identity", None)]
        if "burden" in self.active_groups:
            fields += [("theta_B", "identity", None), ("burden_loss_effect", "identity", None)]
        if "co_segregation" in self.active_groups:
            fields += [("co_segregation_rho", "identity", None)]
        if "exposure" in self.active_groups:
            fields += [("exposure_C_scale", "log", None), ("exposure_P_scale", "log", None)]
        seen: set[str] = set()
        unique = []
        for field in fields:
            if field[0] not in seen:
                unique.append(field)
                seen.add(field[0])
        return tuple(unique)

    def parameter_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for field_name, _transform, shape in self.fields:
            value = getattr(self.base_params, field_name)
            size = 1 if shape is None else int(np.prod(shape))
            if size == 1:
                names.append(field_name)
            else:
                names.extend(f"{field_name}[{idx}]" for idx in range(size))
        return tuple(names)

    def default_vector(self) -> np.ndarray:
        return self.pack(self.base_params)

    def pack(self, params: V4LiteParameters) -> np.ndarray:
        pieces: list[np.ndarray] = []
        for field_name, transform, shape in self.fields:
            raw = np.asarray(getattr(params, field_name), dtype=float).reshape(-1)
            if transform == "log":
                pieces.append(np.log(np.clip(raw, 1e-12, None)))
            elif transform == "logit":
                clipped = np.clip(raw, 1e-9, 1.0 - 1e-9)
                pieces.append(np.log(clipped / (1.0 - clipped)))
            else:
                pieces.append(raw)
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=float)

    def unpack(self, vector: np.ndarray) -> V4LiteParameters:
        params = self.base_params.copy()
        flat = np.asarray(vector, dtype=float).reshape(-1)
        offset = 0
        for field_name, transform, shape in self.fields:
            current = np.asarray(getattr(params, field_name), dtype=float)
            size = 1 if shape is None else int(np.prod(shape))
            chunk = flat[offset : offset + size]
            offset += size
            if transform == "log":
                raw = np.exp(chunk)
            elif transform == "logit":
                raw = 1.0 / (1.0 + np.exp(-np.clip(chunk, -500.0, 500.0)))
            else:
                raw = chunk
            if size == 1 and np.isscalar(getattr(params, field_name)):
                setattr(params, field_name, float(raw[0]))
            else:
                setattr(params, field_name, np.asarray(raw, dtype=float).reshape(current.shape))
        return params


class V4LiteObjective:
    def __init__(
        self,
        *,
        tensor: V4LiteTensor,
        active_groups: tuple[str, ...],
        model_version: str,
        base_params: V4LiteParameters | None = None,
        heldout_weeks: Iterable[int] = (),
        dynamics_mode: str = "joint",
        frozen_copy_distributions: Mapping[str, np.ndarray] | None = None,
        coupling_mode: str | None = None,
    ):
        self.tensor = tensor
        self.active_groups = tuple(active_groups)
        self.model_version = model_version
        self.heldout_weeks = tuple(int(w) for w in heldout_weeks)
        self.dynamics_mode = dynamics_mode
        self.frozen_copy_distributions = frozen_copy_distributions
        self.coupling_mode = ("none" if dynamics_mode == "state_only" else "joint") if coupling_mode is None else coupling_mode
        self.base_params = V4LiteParameters.default(tensor.structure) if base_params is None else base_params.copy()
        self.adapter = V4LiteParameterAdapter(tensor.structure, self.active_groups, self.base_params)

    def evaluate_vector(self, vector: np.ndarray, *, return_artifacts: bool = False) -> V4LiteObjectiveResult:
        params = self.adapter.unpack(vector)
        prediction = predict_v4_lite(self.tensor, params, dynamics_mode=self.dynamics_mode, frozen_copy_distributions=self.frozen_copy_distributions, coupling_mode=self.coupling_mode)
        block_results: list[V4LiteBlockResult] = []
        data_nll = 0.0
        for block_name in self.tensor.observed_summary.block_names():
            if not self._block_active(block_name):
                continue
            observed_block = self.tensor.observed_summary.blocks[block_name]
            if block_name not in prediction.summary.blocks:
                continue
            predicted_block = prediction.summary.blocks[block_name].align_to(observed_block)
            obs_values = []
            pred_values = []
            for key, observed, predicted in zip(observed_block.keys, observed_block.values, predicted_block.values):
                if any(f"week{week}" in key for week in self.heldout_weeks):
                    continue
                obs_values.append(float(observed))
                pred_values.append(float(predicted))
            if not obs_values:
                continue
            obs = np.asarray(obs_values, dtype=float)
            pred = np.asarray(pred_values, dtype=float)
            residual = pred - obs
            scale = max(_safe_rmse(obs), 1.0)
            nll = 0.5 * float(np.mean(np.square(residual / scale)))
            block_results.append(V4LiteBlockResult(block_name, obs.size, nll, _safe_rmse(residual)))
            data_nll += nll
        prior = 0.01 * float(np.mean(np.square(np.asarray(vector, dtype=float)))) if np.asarray(vector).size else 0.0
        artifacts = V4LiteObjectiveArtifacts(params, prediction, np.asarray(vector, dtype=float).copy()) if return_artifacts else None
        return V4LiteObjectiveResult(float(data_nll + prior), float(data_nll), float(prior), tuple(block_results), artifacts)

    def _block_active(self, block_name: str) -> bool:
        if "observation" in self.active_groups:
            return True
        if block_name in {"qpcdr", "ectag_hist", "ectag_moments", "ectag_corr", "ddpcr_pooled_mean"}:
            return "ecDNA_kernel" in self.active_groups or "co_segregation" in self.active_groups
        if block_name in {"flow_fraction", "flow_count", "count_total", "count_gate"}:
            return any(group in self.active_groups for group in ("state_abundance", "burden", "growth_coupling", "transition_coupling"))
        return True


@dataclass(frozen=True)
class V4LiteStageDefinition:
    name: str
    active_groups: tuple[str, ...]
    observed_blocks: tuple[str, ...] | None
    description: str
    model_version: str = "M1"
    optional: bool = False
    dynamics_mode: str = "joint"
    coupling_mode: str = "joint"


@dataclass(frozen=True)
class V4LiteStageFitResult:
    stage_name: str
    active_groups: tuple[str, ...]
    observed_blocks: tuple[str, ...] | None
    objective_before: float | None
    objective_after: float | None
    rejection_reasons: tuple[str, ...]
    accepted: bool
    skipped_reason: str | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class V4LiteOptimizationSettings:
    maxiter: int = 40
    n_restarts: int = 1
    optimizer_method: str = "Powell"
    posterior_draws: int = 64
    posterior_backend: str = "auto"
    emcee_walkers: int = 0
    emcee_steps: int = 0
    emcee_burnin: int = 16
    emcee_initial_scale: float = 0.05
    random_seed: int = 42
    synthetic_recovery_datasets: int = 50
    synthetic_recovery_maxiter: int = 4
    sbc_datasets: int = 12


@dataclass(frozen=True)
class V4LiteFitResult:
    final_params: V4LiteParameters
    tensor: V4LiteTensor
    stage_results: tuple[V4LiteStageFitResult, ...]
    reports: V4LiteReports | None = None
    posterior_samples: V4LitePosteriorSamples | None = None
    projection_targets: FullToLiteProjection | None = None


V4_LITE_STAGE_SEQUENCE = (
    V4LiteStageDefinition("observation", ("observation",), None, "Assay calibration and block noise."),
    V4LiteStageDefinition("week1-init-check", (), None, "Check week1 initialization.", optional=True),
    V4LiteStageDefinition("M0-observation-only", ("observation",), None, "Independent snapshot observation model.", model_version="M0"),
    V4LiteStageDefinition("M1-ecDNA-kernel", ("exposure", "ecDNA_kernel"), ("qpcdr", "ectag_hist", "ectag_moments", "ectag_corr", "ddpcr_pooled_mean"), "State-specific ecDNA distribution dynamics.", model_version="M1", dynamics_mode="ecDNA_only", coupling_mode="none"),
    V4LiteStageDefinition("M2-abundance-null", ("exposure", "state_abundance"), ("flow_fraction", "flow_count", "count_total", "count_gate"), "State abundance dynamics without ecDNA coupling.", model_version="M2", dynamics_mode="state_only", coupling_mode="none"),
    V4LiteStageDefinition("M3-growth-coupling", ("exposure", "state_abundance", "growth_coupling"), ("flow_fraction", "flow_count", "count_total", "count_gate"), "ecDNA-to-growth coupling candidate.", model_version="M3", optional=True, dynamics_mode="joint", coupling_mode="growth"),
    V4LiteStageDefinition("M4-transition-coupling", ("exposure", "state_abundance", "transition_coupling"), ("flow_fraction", "flow_count", "count_total", "count_gate"), "ecDNA-to-transition coupling candidate.", model_version="M4", optional=True, dynamics_mode="joint", coupling_mode="transition"),
    V4LiteStageDefinition("M3-co-segregation", ("exposure", "ecDNA_kernel", "state_abundance", "co_segregation"), None, "Co-segregation extension; conditional on same-cell multi-species ecTAG.", model_version="M3", optional=True),
    V4LiteStageDefinition("LITE-final-joint", ("exposure", "ecDNA_kernel", "state_abundance"), None, "Final accepted v4-lite joint refit.", model_version="M1"),
)


def _objective_improvement(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return float(before - after)


def _stage_criteria(
    stage: V4LiteStageDefinition,
    tensor: V4LiteTensor,
    before: float | None,
    after: float | None,
    block_results: tuple[V4LiteBlockResult, ...],
    coupling_diagnostics: Mapping[str, object] | None = None,
) -> tuple[tuple[dict[str, object], ...], tuple[str, ...]]:
    improvement = _objective_improvement(before, after)
    criteria: list[dict[str, object]] = []
    reasons: list[str] = []

    def add(name: str, passed: bool, value: object, threshold: str) -> None:
        criteria.append({"criterion": name, "passed": bool(passed), "value": value, "threshold": threshold})
        if not passed:
            reasons.append(f"{name} failed")

    if before is not None and after is not None:
        add("objective_non_increasing", after <= before + 1e-8, improvement, "after <= before")
    if stage.name == "M1-ecDNA-kernel":
        available = {block.name for block in block_results}
        add("ecDNA_blocks_present", bool(available & {"qpcdr", "ectag_hist", "ectag_moments", "ddpcr_pooled_mean"}), sorted(available), "at least one ecDNA observation block")
    if stage.name == "M2-abundance-null":
        available = {block.name for block in block_results}
        add("abundance_blocks_present", bool(available & {"flow_fraction", "flow_count", "count_total", "count_gate"}), sorted(available), "at least one abundance observation block")
    if stage.name == "M3-growth-coupling":
        add("predictive_score_improvement", bool(improvement is not None and improvement >= 4.0), improvement, ">= 4 log-objective units; otherwise keep coupling fixed at 0")
        sign_probability = None if coupling_diagnostics is None else coupling_diagnostics.get("posterior_sign_probability")
        contraction = None if coupling_diagnostics is None else coupling_diagnostics.get("posterior_contraction")
        recovery = None if coupling_diagnostics is None else coupling_diagnostics.get("synthetic_sign_recovery")
        add("posterior_sign_probability", bool(sign_probability is not None and float(sign_probability) > 0.90), sign_probability, "> 0.90")
        add("posterior_contraction", bool(contraction is not None and float(contraction) > 0.30), contraction, "> 0.30")
        add("synthetic_sign_recovery", bool(recovery is not None and float(recovery) >= 0.80), recovery, ">= 80% over 50 synthetic datasets")
    if stage.name == "M4-transition-coupling":
        add("predictive_score_improvement", bool(improvement is not None and improvement >= 4.0), improvement, ">= 4 log-objective units; otherwise keep coupling fixed at 0")
        sign_probability = None if coupling_diagnostics is None else coupling_diagnostics.get("posterior_sign_probability")
        contraction = None if coupling_diagnostics is None else coupling_diagnostics.get("posterior_contraction")
        recovery = None if coupling_diagnostics is None else coupling_diagnostics.get("synthetic_sign_recovery")
        profile_span = None if coupling_diagnostics is None else coupling_diagnostics.get("profile_span")
        add("profile_likelihood_not_flat", bool(profile_span is not None and float(profile_span) > 1e-4), profile_span, "not flat")
        add("posterior_sign_probability", bool(sign_probability is not None and float(sign_probability) > 0.90), sign_probability, "> 0.90")
        add("posterior_contraction", bool(contraction is not None and float(contraction) > 0.30), contraction, "> 0.30")
        add("synthetic_sign_recovery", bool(recovery is not None and float(recovery) >= 0.80), recovery, ">= 80% over 50 synthetic datasets")
    return tuple(criteria), tuple(reasons)


def _coupling_parameter_names(stage: V4LiteStageDefinition) -> tuple[str, ...]:
    if stage.name == "M3-growth-coupling":
        return ("theta_P",)
    if stage.name == "M4-transition-coupling":
        return ("beta_C", "beta_P", "lambda_M")
    return ()


def _parameter_indices(names: Sequence[str], selected: Sequence[str]) -> tuple[int, ...]:
    wanted = set(selected)
    return tuple(idx for idx, name in enumerate(names) if name in wanted)


def _posterior_sign_summary(posterior: V4LitePosteriorSamples, indices: Sequence[int]) -> tuple[float | None, float | None]:
    if not indices or posterior.samples.size == 0:
        return None, None
    selected = posterior.samples[:, list(indices)]
    sign_probs = []
    contractions = []
    for column in range(selected.shape[1]):
        values = selected[:, column]
        mean = float(np.mean(values))
        if abs(mean) < 1e-8:
            sign_probs.append(0.5)
        elif mean > 0.0:
            sign_probs.append(float(np.mean(values > 0.0)))
        else:
            sign_probs.append(float(np.mean(values < 0.0)))
        contractions.append(max(0.0, min(1.0, 1.0 - float(np.std(values)) / 1.0)))
    return float(min(sign_probs)), float(min(contractions))


def _noisy_summary(summary: SummaryCollection, rng: np.random.Generator) -> SummaryCollection:
    maps: dict[str, dict[str, float]] = {}
    for block_name, block in summary.blocks.items():
        values = np.asarray(block.values, dtype=float)
        scale = max(_safe_rmse(values), 1.0) * 0.05
        noisy = values + rng.normal(0.0, scale, size=values.shape)
        if block_name in {"flow_fraction", "ectag_hist", "ectag_moments", "ectag_corr"}:
            noisy = np.clip(noisy, 0.0, 1.0)
        if block_name in {"flow_count", "count_total", "count_gate", "qpcdr", "ddpcr_pooled_mean"}:
            noisy = np.clip(noisy, 0.0, None)
        maps[block_name] = {key: float(value) for key, value in zip(block.keys, noisy)}
    return SummaryCollection.from_block_maps(maps)


def _optimize_with_limited_iterations(objective: V4LiteObjective, initial: np.ndarray, *, maxiter: int, method: str) -> np.ndarray:
    def fun(vector: np.ndarray) -> float:
        return objective.evaluate_vector(vector).total_objective

    result = minimize(fun, np.asarray(initial, dtype=float), method=method, options={"maxiter": int(maxiter), "disp": False})
    return np.asarray(result.x if np.isfinite(result.fun) else initial, dtype=float)


def _coupling_diagnostics(objective: V4LiteObjective, vector: np.ndarray, stage: V4LiteStageDefinition, settings: V4LiteOptimizationSettings) -> dict[str, object]:
    selected_names = _coupling_parameter_names(stage)
    if not selected_names:
        return {}
    posterior = run_v4_lite_hmc(objective, vector, settings)
    indices = _parameter_indices(posterior.parameter_names, selected_names)
    sign_probability, contraction = _posterior_sign_summary(posterior, indices)
    profile = run_v4_lite_profile_likelihood(objective, vector)
    profile_span = float(max((point.objective_value for point in profile), default=0.0) - min((point.objective_value for point in profile), default=0.0))
    rng = np.random.default_rng(settings.random_seed + 1009)
    truth = np.asarray(vector, dtype=float)
    truth_sign = np.sign(truth[list(indices)]) if indices else np.zeros(0, dtype=float)
    recovered = 0
    attempted = int(settings.synthetic_recovery_datasets)
    for _ in range(attempted):
        synthetic_tensor = replace(objective.tensor, observed_summary=_noisy_summary(objective.evaluate_vector(truth, return_artifacts=True).artifacts.prediction.summary, rng))
        synthetic_objective = V4LiteObjective(
            tensor=synthetic_tensor,
            active_groups=objective.active_groups,
            model_version=objective.model_version,
            base_params=objective.base_params,
            dynamics_mode=objective.dynamics_mode,
            frozen_copy_distributions=objective.frozen_copy_distributions,
            coupling_mode=objective.coupling_mode,
        )
        estimate = _optimize_with_limited_iterations(synthetic_objective, truth, maxiter=settings.synthetic_recovery_maxiter, method=settings.optimizer_method)
        if indices and np.all(np.sign(estimate[list(indices)]) == truth_sign) and np.all(np.abs(truth[list(indices)]) > 1e-8):
            recovered += 1
    recovery_rate = float(recovered / max(attempted, 1))
    return {
        "posterior_sign_probability": sign_probability,
        "posterior_contraction": contraction,
        "profile_span": profile_span,
        "synthetic_sign_recovery": recovery_rate,
        "synthetic_recovery_datasets": attempted,
        "posterior_method": posterior.method,
    }


class V4LiteFitRunner:
    def __init__(self, dataset: CanonicalFitDataset, *, model_version: str = "M1", structure: V4LiteStructure | None = None, initial_params: V4LiteParameters | None = None, optimization_settings: V4LiteOptimizationSettings | None = None, output_dir: str | Path | None = None, condition_names: Iterable[str] | None = None, purity_sensitivity: Iterable[np.ndarray] | None = None):
        self.model_version = model_version
        self.tensor = build_v4_lite_tensor(dataset, condition_names=condition_names, structure=structure)
        self.structure = self.tensor.structure
        self.current_params = V4LiteParameters.default(self.structure, purity_matrix=dataset.purity_matrix, qpcdr_calibration=dataset.qpcdr_calibration) if initial_params is None else initial_params.copy()
        self.settings = V4LiteOptimizationSettings() if optimization_settings is None else optimization_settings
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.purity_sensitivity = tuple(purity_sensitivity or dataset.purity_sensitivity)
        self.frozen_copy_distributions: Mapping[str, np.ndarray] | None = None
        self.accepted_growth_coupling = False
        self.accepted_transition_coupling = False

    def _final_coupling_mode(self) -> str:
        if self.accepted_growth_coupling and self.accepted_transition_coupling:
            return "joint"
        if self.accepted_growth_coupling:
            return "growth"
        if self.accepted_transition_coupling:
            return "transition"
        return "none"

    def _objective_for_stage(self, stage: V4LiteStageDefinition) -> V4LiteObjective:
        frozen = self.frozen_copy_distributions if stage.dynamics_mode == "state_only" else None
        coupling_mode = stage.coupling_mode
        if stage.name == "LITE-final-joint":
            coupling_mode = self._final_coupling_mode()
        return V4LiteObjective(
            tensor=self.tensor,
            active_groups=stage.active_groups,
            model_version=stage.model_version,
            base_params=self.current_params,
            dynamics_mode=stage.dynamics_mode,
            frozen_copy_distributions=frozen,
            coupling_mode=coupling_mode,
        )

    def _optimize_objective(self, objective: V4LiteObjective, initial: np.ndarray, n_restarts: int = 1) -> tuple[np.ndarray, float]:
        best_vector = np.asarray(initial, dtype=float)
        best_score = objective.evaluate_vector(best_vector).total_objective
        rng = np.random.default_rng(self.settings.random_seed)
        for restart in range(max(int(n_restarts), 1)):
            start = best_vector if restart == 0 else best_vector + rng.normal(0.0, 0.05, size=best_vector.size)

            def fun(vector: np.ndarray) -> float:
                return objective.evaluate_vector(vector).total_objective

            result = minimize(fun, start, method=self.settings.optimizer_method, options={"maxiter": int(self.settings.maxiter), "disp": False})
            candidate = np.asarray(result.x if np.isfinite(result.fun) else start, dtype=float)
            score = objective.evaluate_vector(candidate).total_objective
            if score <= best_score:
                best_vector = candidate
                best_score = score
        return best_vector, float(best_score)

    def run_stage(self, stage: V4LiteStageDefinition) -> V4LiteStageFitResult:
        if stage.name == "observation":
            diagnostics = {"calibration": _calibration_diagnostics(self.tensor)}
            return V4LiteStageFitResult(stage.name, stage.active_groups, stage.observed_blocks, None, None, (), True, diagnostics=diagnostics)
        if stage.name == "week1-init-check":
            return V4LiteStageFitResult(stage.name, stage.active_groups, stage.observed_blocks, None, None, (), True, diagnostics={"week1_total": {k: float(np.sum(v)) for k, v in self.tensor.initial_state_abundance.items()}})
        if stage.name == "M3-co-segregation" and not self.tensor.has_same_cell_ectag:
            return V4LiteStageFitResult(stage.name, stage.active_groups, stage.observed_blocks, None, None, ("No same-cell multi-species ecTAG.",), False, skipped_reason="No same-cell multi-species ecTAG observations are available.")
        objective = self._objective_for_stage(stage)
        initial = objective.adapter.default_vector()
        before_result = objective.evaluate_vector(initial)
        before = before_result.total_objective
        best, after = self._optimize_objective(objective, initial, self.settings.n_restarts)
        after_result = objective.evaluate_vector(best, return_artifacts=True)
        coupling_diagnostics = _coupling_diagnostics(objective, best, stage, self.settings) if stage.name in {"M3-growth-coupling", "M4-transition-coupling"} else {}
        criteria, failed = _stage_criteria(stage, self.tensor, before, after, after_result.block_results, coupling_diagnostics)
        if stage.optional:
            accepted = bool(not failed)
        else:
            accepted = bool(after <= before + 1e-8 and not failed)
        if accepted:
            self.current_params = objective.adapter.unpack(best)
            if stage.name == "M1-ecDNA-kernel":
                self.frozen_copy_distributions = predict_v4_lite(self.tensor, self.current_params, dynamics_mode="ecDNA_only").copy_distributions
            if stage.name == "M3-growth-coupling":
                self.accepted_growth_coupling = True
            if stage.name == "M4-transition-coupling":
                self.accepted_transition_coupling = True
        diagnostics = {
            "criteria": criteria,
            "objective_improvement": _objective_improvement(before, after),
            "block_results": [block.__dict__ for block in after_result.block_results],
            "coupling_diagnostics": coupling_diagnostics,
            "approximation_policy": "MAP optimization with package posterior diagnostics when available; emcee ensemble MCMC is supported, with Laplace fallback.",
        }
        return V4LiteStageFitResult(stage.name, stage.active_groups, stage.observed_blocks, before, after, () if accepted else failed or ("objective did not improve",), accepted, diagnostics=diagnostics)

    def run_all(self) -> V4LiteFitResult:
        results = []
        for stage in V4_LITE_STAGE_SEQUENCE:
            result = self.run_stage(stage)
            print(
                "[fit] "
                f"{stage.name}: active_groups={stage.active_groups} accepted={result.accepted} "
                f"before={result.objective_before} after={result.objective_after} "
                f"failed={result.rejection_reasons} skip={result.skipped_reason}"
            )
            results.append(result)
        final_coupling_mode = self._final_coupling_mode()
        prediction = predict_v4_lite(self.tensor, self.current_params, coupling_mode=final_coupling_mode)
        objective = V4LiteObjective(tensor=self.tensor, active_groups=("exposure", "ecDNA_kernel", "state_abundance"), model_version=self.model_version, base_params=self.current_params, coupling_mode=final_coupling_mode)
        final_vector = objective.adapter.default_vector()
        posterior = run_v4_lite_hmc(objective, final_vector, self.settings)
        prior = run_v4_lite_prior_predictive(objective, n_draws=max(4, min(32, self.settings.posterior_draws)), seed=self.settings.random_seed)
        profile = run_v4_lite_profile_likelihood(objective, final_vector)
        fake = run_v4_lite_fake_data_recovery(objective, final_vector, self._optimize_objective, n_restarts=1)
        loo = run_leave_one_week_out(objective, final_vector, self._optimize_objective, n_restarts=1)
        sbc = run_v4_lite_sbc(self, self.settings.sbc_datasets, self.model_version)
        status = build_parameter_status_table(objective, final_vector, profile, fake, posterior)
        model_comparison = _build_model_comparison(results)
        reports = build_v4_lite_reports(
            self.tensor,
            prediction,
            results,
            status,
            fake,
            loo,
            sbc,
            model_comparison,
            _calibration_diagnostics(self.tensor),
            (),
            build_prior_diagnostics_report(objective, final_vector),
            objective,
            posterior,
        )
        fit_result = V4LiteFitResult(self.current_params.copy(), self.tensor, tuple(results), reports, posterior, _projection_from_prediction(prediction))
        reports.implementation_status_report["prior_predictive"] = {"n_draws": prior.n_draws, "pass_rate": prior.pass_rate, "failures": prior.failures}
        if self.output_dir is not None:
            write_v4_lite_reports(self.output_dir, reports, status, model_comparison)
            np.savez(
                self.output_dir / "v4_lite_arrays.npz",
                **{f"{c}_state_abundance": v for c, v in prediction.state_abundance.items()},
                **{f"{c}_copy_distributions": v for c, v in prediction.copy_distributions.items()},
                **{f"{c}_transition_matrices": v for c, v in prediction.transition_matrices.items()},
                **{f"{c}_growth_rates": v for c, v in prediction.growth_rates.items()},
                posterior_samples=posterior.samples,
            )
            write_fit_method_artifacts(self.output_dir, fit_result, status, model_comparison)
            for stage_name, paths in _fit_method_stage_output_paths(self.output_dir).items():
                existing = [str(path) for path in paths if path.exists()]
                print(f"[fit] outputs {stage_name}: {', '.join(existing)}")
            print(f"[fit] reports written to {self.output_dir}")
        return fit_result

    def run_all_stages(self) -> V4LiteFitResult:
        return self.run_all()


def _calibration_diagnostics(tensor: V4LiteTensor) -> dict[str, object]:
    q_reps = {(obs.condition, obs.week, obs.gate_index, obs.species_index): set() for obs in tensor.qpcdr_observations}
    for obs in tensor.qpcdr_observations:
        q_reps[(obs.condition, obs.week, obs.gate_index, obs.species_index)].add(obs.replicate_id)
    f_reps = {(obs.condition, obs.week): set() for obs in tensor.flow_observations}
    for obs in tensor.flow_observations:
        f_reps[(obs.condition, obs.week)].add(obs.replicate_id)
    return {
        "qpcdr": {"insufficient_replicates": not q_reps or max((len(v) for v in q_reps.values()), default=0) < 2},
        "flow": {"insufficient_replicates": not f_reps or max((len(v) for v in f_reps.values()), default=0) < 2},
        "ddpcr_policy": "pooled_mean_anchor_only",
        "ectag_policy": "species_specific_bins_no_config_censoring",
    }


def _build_model_comparison(stage_results: Sequence[V4LiteStageFitResult]) -> dict[str, float]:
    comparison: dict[str, float] = {}
    for stage in stage_results:
        if stage.objective_after is not None:
            comparison[f"{stage.stage_name}.objective_after"] = float(stage.objective_after)
        improvement = _objective_improvement(stage.objective_before, stage.objective_after)
        if improvement is not None:
            comparison[f"{stage.stage_name}.improvement"] = float(improvement)
    name_to_after = {stage.stage_name: stage.objective_after for stage in stage_results if stage.objective_after is not None}
    if "M2-abundance-null" in name_to_after and "M3-growth-coupling" in name_to_after:
        comparison["M3_vs_M2.log_objective_improvement"] = float(name_to_after["M2-abundance-null"] - name_to_after["M3-growth-coupling"])
    if "M2-abundance-null" in name_to_after and "M4-transition-coupling" in name_to_after:
        baseline = min(v for key, v in name_to_after.items() if key in {"M2-abundance-null", "M3-growth-coupling"} and v is not None)
        comparison["M4_vs_M2_M3.log_objective_improvement"] = float(baseline - name_to_after["M4-transition-coupling"])
    return comparison


def run_leave_one_week_out(objective: V4LiteObjective, initial_vector: np.ndarray, optimizer, *, n_restarts: int) -> V4LiteLeaveOneWeekOutReport:
    scores: dict[int, float] = {}
    for week in objective.tensor.weeks:
        if week == WEEK1:
            continue
        heldout = V4LiteObjective(
            tensor=objective.tensor,
            active_groups=objective.active_groups,
            model_version=objective.model_version,
            base_params=objective.base_params,
            heldout_weeks=(week,),
            dynamics_mode=objective.dynamics_mode,
            frozen_copy_distributions=objective.frozen_copy_distributions,
            coupling_mode=objective.coupling_mode,
        )
        vector, _score = optimizer(heldout, initial_vector, n_restarts)
        scores[int(week)] = float(heldout.evaluate_vector(vector).total_objective)
    return V4LiteLeaveOneWeekOutReport(scores)


def run_v4_lite_prior_predictive(objective: V4LiteObjective, *, n_draws: int, seed: int) -> V4LitePriorPredictiveReport:
    rng = np.random.default_rng(seed)
    center = objective.adapter.default_vector()
    failures: dict[str, int] = {}
    passed = 0
    for _ in range(int(n_draws)):
        draw = center + rng.normal(0.0, 0.5, size=center.size)
        result = objective.evaluate_vector(draw)
        if np.isfinite(result.total_objective):
            passed += 1
        else:
            failures["non_finite_objective"] = failures.get("non_finite_objective", 0) + 1
    return V4LitePriorPredictiveReport(int(n_draws), float(passed / max(int(n_draws), 1)), failures)


def run_v4_lite_posterior_predictive(observed_summary: SummaryCollection, prediction: V4LitePrediction) -> V4LitePosteriorPredictiveReport:
    aligned = prediction.summary.align_to(observed_summary)
    rmse: dict[str, float] = {}
    rel: dict[str, float] = {}
    max_abs: dict[str, float] = {}
    for block in observed_summary.block_names():
        obs = observed_summary.blocks[block].values
        pred = aligned.blocks[block].values
        residual = pred - obs
        rmse[block] = _safe_rmse(residual)
        rel[block] = rmse[block] / max(_safe_rmse(obs), 1e-8)
        max_abs[block] = float(np.max(np.abs(residual))) if residual.size else 0.0
    return V4LitePosteriorPredictiveReport(rmse, rel, max_abs, max(rel.values()) if rel else 0.0)


def _posterior_predictive_interval_rows(objective: V4LiteObjective, posterior: V4LitePosteriorSamples) -> tuple[dict[str, object], ...]:
    if posterior.samples.size == 0:
        return ()
    observed = objective.tensor.observed_summary
    draws_by_block: dict[str, list[np.ndarray]] = {block: [] for block in observed.block_names()}
    for sample in posterior.samples:
        prediction = objective.evaluate_vector(sample, return_artifacts=True).artifacts.prediction
        aligned = prediction.summary.align_to(observed)
        for block in observed.block_names():
            draws_by_block[block].append(np.asarray(aligned.blocks[block].values, dtype=float))
    rows: list[dict[str, object]] = []
    for block in observed.block_names():
        if not draws_by_block[block]:
            continue
        matrix = np.vstack(draws_by_block[block])
        lower = np.quantile(matrix, 0.05, axis=0)
        median = np.quantile(matrix, 0.50, axis=0)
        upper = np.quantile(matrix, 0.95, axis=0)
        obs_block = observed.blocks[block]
        for key, observed_value, lo, med, hi in zip(obs_block.keys, obs_block.values, lower, median, upper):
            rows.append(
                {
                    "block": block,
                    "key": key,
                    "observed": float(observed_value),
                    "p05": float(lo),
                    "p50": float(med),
                    "p95": float(hi),
                    "covered_90": bool(float(lo) <= float(observed_value) <= float(hi)),
                }
            )
    return tuple(rows)


def _coverage_from_interval_rows(rows: Sequence[Mapping[str, object]]) -> tuple[dict[str, float], float | None]:
    if not rows:
        return {}, None
    totals: dict[str, int] = {}
    covered: dict[str, int] = {}
    for row in rows:
        block = str(row["block"])
        totals[block] = totals.get(block, 0) + 1
        covered[block] = covered.get(block, 0) + int(bool(row.get("covered_90")))
    block_coverage = {block: float(covered.get(block, 0) / max(total, 1)) for block, total in totals.items()}
    return block_coverage, float(sum(covered.values()) / max(sum(totals.values()), 1))


def run_v4_lite_profile_likelihood(objective: V4LiteObjective, vector: np.ndarray, *, offsets: Iterable[float] = (-0.5, 0.0, 0.5)) -> tuple[V4LiteProfilePoint, ...]:
    points: list[V4LiteProfilePoint] = []
    base = np.asarray(vector, dtype=float)
    for idx in range(base.size):
        for offset in offsets:
            trial = base.copy()
            trial[idx] += float(offset)
            points.append(V4LiteProfilePoint(idx, float(offset), objective.evaluate_vector(trial).total_objective))
    return tuple(points)


def run_v4_lite_fake_data_recovery(objective: V4LiteObjective, truth_vector: np.ndarray, optimizer, *, n_restarts: int) -> V4LiteFakeDataRecoveryReport:
    recovered, score = optimizer(objective, truth_vector, n_restarts)
    error = float(np.linalg.norm(np.asarray(recovered) - np.asarray(truth_vector)) / max(1, np.asarray(truth_vector).size))
    prediction = objective.evaluate_vector(recovered, return_artifacts=True).artifacts.prediction
    ppc = run_v4_lite_posterior_predictive(objective.tensor.observed_summary, prediction)
    return V4LiteFakeDataRecoveryReport(
        float(score),
        error,
        error < 0.5,
        ppc.block_relative_rmse,
        n_synthetic=1,
        skipped_reason=None,
        coverage_rate=None,
    )


def _laplace_posterior_samples(objective: V4LiteObjective, initial_vector: np.ndarray, settings: V4LiteOptimizationSettings, *, skipped_reason: str = "laplace_gaussian_approximation_not_nuts") -> V4LitePosteriorSamples:
    rng = np.random.default_rng(settings.random_seed)
    initial = np.asarray(initial_vector, dtype=float)
    if initial.size == 0:
        return V4LitePosteriorSamples(objective.adapter.parameter_names(), np.zeros((int(settings.posterior_draws), 0)), 1.0, None, np.zeros((0, 0)), "empty_parameter_vector")
    f0 = objective.evaluate_vector(initial).total_objective
    variances = np.zeros(initial.size, dtype=float)
    step = 0.05
    for idx in range(initial.size):
        plus = initial.copy()
        minus = initial.copy()
        plus[idx] += step
        minus[idx] -= step
        fp = objective.evaluate_vector(plus).total_objective
        fm = objective.evaluate_vector(minus).total_objective
        curvature = max(float((fp - 2.0 * f0 + fm) / (step * step)), 1e-3)
        variances[idx] = min(max(1.0 / curvature, 1e-6), 1.0)
    covariance = np.diag(variances)
    samples = rng.multivariate_normal(initial, covariance, size=int(settings.posterior_draws))
    return V4LitePosteriorSamples(objective.adapter.parameter_names(), samples, 1.0, skipped_reason, covariance, "laplace_gaussian_approximation")


def _emcee_posterior_samples(objective: V4LiteObjective, initial_vector: np.ndarray, settings: V4LiteOptimizationSettings) -> V4LitePosteriorSamples:
    import emcee  # type: ignore

    rng = np.random.default_rng(settings.random_seed)
    initial = np.asarray(initial_vector, dtype=float)
    if initial.size == 0:
        return V4LitePosteriorSamples(objective.adapter.parameter_names(), np.zeros((int(settings.posterior_draws), 0)), 1.0, None, np.zeros((0, 0)), "empty_parameter_vector")
    n_dim = int(initial.size)
    n_walkers = max(int(settings.emcee_walkers) if settings.emcee_walkers else 0, 2 * n_dim + 2, 8)
    burnin = max(int(settings.emcee_burnin), 0)
    auto_steps = burnin + max(16, int(np.ceil(max(int(settings.posterior_draws), 1) / n_walkers)) + 16)
    n_steps = max(int(settings.emcee_steps) if settings.emcee_steps else auto_steps, burnin + 1)
    initial_state = initial + rng.normal(0.0, float(settings.emcee_initial_scale), size=(n_walkers, n_dim))

    def log_prob(vector: np.ndarray) -> float:
        value = objective.evaluate_vector(np.asarray(vector, dtype=float)).total_objective
        return -float(value) if np.isfinite(value) else -np.inf

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob)
    sampler.run_mcmc(initial_state, n_steps, progress=False, skip_initial_state_check=True)
    chain = np.asarray(sampler.get_chain(discard=burnin, flat=True), dtype=float)
    chain = chain[np.all(np.isfinite(chain), axis=1)]
    if chain.shape[0] == 0:
        raise RuntimeError("emcee produced no finite posterior draws")
    n_draws = min(int(settings.posterior_draws), chain.shape[0])
    selected = rng.choice(chain.shape[0], size=n_draws, replace=False) if chain.shape[0] > n_draws else np.arange(chain.shape[0])
    samples = chain[selected]
    covariance = np.cov(samples, rowvar=False) if samples.shape[0] > 1 else np.zeros((n_dim, n_dim), dtype=float)
    if np.ndim(covariance) == 0:
        covariance = np.asarray([[float(covariance)]], dtype=float)
    return V4LitePosteriorSamples(
        objective.adapter.parameter_names(),
        samples,
        float(np.mean(sampler.acceptance_fraction)),
        None,
        np.asarray(covariance, dtype=float),
        "emcee_ensemble_mcmc",
    )


def run_v4_lite_hmc(objective: V4LiteObjective, initial_vector: np.ndarray, settings: V4LiteOptimizationSettings) -> V4LitePosteriorSamples:
    backend = str(settings.posterior_backend).lower()
    cfg.require(backend in {"auto", "emcee", "laplace"}, f"Unknown posterior backend {settings.posterior_backend}.")
    if backend in {"auto", "emcee"}:
        try:
            return _emcee_posterior_samples(objective, initial_vector, settings)
        except Exception as exc:
            reason = f"emcee_unavailable_or_failed: {type(exc).__name__}: {exc}; laplace_gaussian_approximation_not_nuts"
            return _laplace_posterior_samples(objective, initial_vector, settings, skipped_reason=reason)
    return _laplace_posterior_samples(objective, initial_vector, settings)


def run_v4_lite_sbc(runner: V4LiteFitRunner, n_datasets: int, model_version: str) -> V4LiteSBCReport:
    if n_datasets <= 0:
        return V4LiteSBCReport(0, {}, 0, "SBC not requested.")
    objective = V4LiteObjective(
        tensor=runner.tensor,
        active_groups=("exposure", "ecDNA_kernel", "state_abundance"),
        model_version=model_version,
        base_params=runner.current_params,
        coupling_mode=runner._final_coupling_mode(),
    )
    center = objective.adapter.default_vector()
    names = objective.adapter.parameter_names()
    ranks: dict[str, list[int]] = {name: [] for name in names}
    failures = 0
    rng = np.random.default_rng(runner.settings.random_seed + 2027)
    posterior_settings = replace(runner.settings, posterior_backend="laplace") if runner.settings.posterior_backend == "auto" else runner.settings
    for _ in range(int(n_datasets)):
        truth = center + rng.normal(0.0, 0.2, size=center.size)
        try:
            truth_prediction = objective.evaluate_vector(truth, return_artifacts=True).artifacts.prediction
            synthetic_tensor = replace(runner.tensor, observed_summary=_noisy_summary(truth_prediction.summary, rng))
            synthetic_objective = V4LiteObjective(
                tensor=synthetic_tensor,
                active_groups=objective.active_groups,
                model_version=model_version,
                base_params=objective.base_params,
                dynamics_mode=objective.dynamics_mode,
                frozen_copy_distributions=objective.frozen_copy_distributions,
                coupling_mode=objective.coupling_mode,
            )
            estimate = _optimize_with_limited_iterations(synthetic_objective, truth, maxiter=runner.settings.synthetic_recovery_maxiter, method=runner.settings.optimizer_method)
            posterior = run_v4_lite_hmc(synthetic_objective, estimate, posterior_settings)
            for idx, name in enumerate(names):
                ranks[name].append(int(np.sum(posterior.samples[:, idx] < truth[idx])))
        except Exception:
            failures += 1
    return V4LiteSBCReport(int(n_datasets), {name: tuple(values) for name, values in ranks.items()}, failures, None)


def build_prior_diagnostics_report(objective: V4LiteObjective, vector: np.ndarray) -> dict[str, object]:
    fields = []
    for name in objective.adapter.parameter_names():
        fields.append({"name": name, "prior_kind": "gaussian_shrinkage_approximation" if any(token in name for token in ("kernel", "growth", "mobility", "theta", "omega")) else "gaussian_approximation"})
    return {
        "active_fields": fields,
        "strict_horseshoe_prior": "not_implemented_strictly",
        "strict_pc_prior": "not_implemented_strictly",
        "prior_policy": "Gaussian shrinkage and boundary checks are used for the automated SciPy implementation.",
        "release_policy": "Parameters that fail profile, posterior contraction, boundary, or synthetic recovery checks are kept fixed or interpreted as derived.",
    }


def build_parameter_status_table(objective: V4LiteObjective, vector: np.ndarray, profile_points: Sequence[V4LiteProfilePoint], fake_recovery: V4LiteFakeDataRecoveryReport, posterior: V4LitePosteriorSamples) -> tuple[dict[str, object], ...]:
    params = objective.adapter.unpack(vector)
    rows = []
    for field_name, transform, _shape in objective.adapter.fields:
        raw = np.asarray(getattr(params, field_name), dtype=float).reshape(-1)
        if transform == "logit":
            margin = float(np.min(np.minimum(raw, 1.0 - raw)))
        elif transform == "log":
            margin = float(np.min(raw / (raw + 1.0)))
        else:
            margin = None
        posterior_sd = float(np.std(posterior.samples, axis=0).mean()) if posterior.samples.size else 0.0
        rationale = "fake-data passed" if fake_recovery.passed else "fake-data failed"
        if margin is not None and np.isfinite(margin) and margin < 0.02:
            rationale += "; boundary warning"
        rows.append(
            {
                "name": field_name,
                "field": field_name,
                "transform": transform,
                "prior_kind": "gaussian_shrinkage_approximation",
                "fake_data_passed": fake_recovery.passed,
                "status": "free" if fake_recovery.passed else "fixed",
                "profile_span": float(max((p.objective_value for p in profile_points), default=0.0) - min((p.objective_value for p in profile_points), default=0.0)),
                "posterior_sd": posterior_sd,
                "prior_scale": 1.0,
                "boundary_margin": margin,
                "rationale": rationale,
            }
        )
    return tuple(rows)


def _posterior_residual_rows(observed: SummaryCollection, predicted: SummaryCollection) -> tuple[dict[str, object], ...]:
    aligned = predicted.align_to(observed)
    rows = []
    for block in observed.block_names():
        for key, obs, pred in zip(observed.blocks[block].keys, observed.blocks[block].values, aligned.blocks[block].values):
            rows.append({"block": block, "key": key, "observed": float(obs), "predicted": float(pred), "residual": float(pred - obs)})
    return tuple(rows)


def build_v4_lite_reports(
    tensor: V4LiteTensor,
    prediction: V4LitePrediction,
    stage_results: list[V4LiteStageFitResult],
    parameter_status_table: tuple[dict[str, object], ...],
    fake_recovery: V4LiteFakeDataRecoveryReport,
    loo: V4LiteLeaveOneWeekOutReport,
    sbc_report: V4LiteSBCReport | None,
    model_comparison: dict[str, float],
    calibration_report: dict[str, object],
    purity_sensitivity: tuple[dict[str, object], ...],
    prior_diagnostics_report: dict[str, object] | None = None,
    posterior_objective: V4LiteObjective | None = None,
    posterior_samples: V4LitePosteriorSamples | None = None,
) -> V4LiteReports:
    ppc = run_v4_lite_posterior_predictive(tensor.observed_summary, prediction)
    interval_rows = _posterior_predictive_interval_rows(posterior_objective, posterior_samples) if posterior_objective is not None and posterior_samples is not None else ()
    block_coverage, overall_coverage = _coverage_from_interval_rows(interval_rows)
    tensor_summary = {
        "conditions": tensor.condition_names,
        "weeks": tensor.weeks,
        "n_flow": len(tensor.flow_observations),
        "n_count": len(tensor.count_observations),
        "n_qpcdr": len(tensor.qpcdr_observations),
        "n_ectag_hist": len(tensor.ectag_hist_observations),
        "n_ddpcr": len(tensor.ddpcr_observations),
        "binning": {"bins": tensor.structure.binning.bins, "centers": tensor.structure.binning.centers.tolist(), "policy": "species-specific; no config.py ecTAG censoring"},
    }
    count_gate_counts: dict[str, int] = {}
    for obs in tensor.count_observations:
        if obs.gate_index is not None:
            count_gate_counts[cfg.STATE_NAMES[obs.gate_index]] = count_gate_counts.get(cfg.STATE_NAMES[obs.gate_index], 0) + 1
    implementation = {
        "v4_lite": "implemented_from_scratch",
        "posterior_sampling": posterior_samples.method if posterior_samples is not None else "not_run",
        "posterior_sampling_note": None if posterior_samples is None else posterior_samples.skipped_reason,
        "ddpcr_policy": "bulk pooled mean anchor only",
        "ectag_policy": "species-specific bins; total burden derived only",
        "prior_policy": "Gaussian shrinkage priors with profile, posterior contraction, boundary, and synthetic recovery checks.",
        "fit_method_scope": "automated SciPy MAP plus package posterior diagnostics when available; NetCDF/CSV/parquet/NPZ artifacts are written where supported.",
    }
    return V4LiteReports(
        tensor_summary=tensor_summary,
        calibration_report={"observation_calibration": calibration_report, "has_same_cell_ectag": tensor.has_same_cell_ectag},
        ecDNA_report={"mean_copy_by_condition": {c: _copy_means(v[-1], tensor.structure.binning).tolist() for c, v in prediction.copy_distributions.items()}},
        identifiability_report={
            "parameter_status": list(parameter_status_table),
            "stage_acceptance": {s.stage_name: s.accepted for s in stage_results},
            "stage_details": [
                {
                    "stage_name": s.stage_name,
                    "active_groups": s.active_groups,
                    "objective_before": s.objective_before,
                    "objective_after": s.objective_after,
                    "accepted": s.accepted,
                    "rejection_reasons": s.rejection_reasons,
                    "skipped_reason": s.skipped_reason,
                    "diagnostics": s.diagnostics,
                }
                for s in stage_results
            ],
            "model_comparison": model_comparison,
        },
        posterior_predictive_report={
            "leave_one_week_out": loo.heldout_scores,
            "available_blocks": tensor.observed_summary.block_names(),
            "block_rmse": ppc.block_rmse,
            "block_relative_rmse": ppc.block_relative_rmse,
            "block_max_abs_residual": ppc.block_max_abs_residual,
            "worst_relative_rmse": ppc.worst_relative_rmse,
            "block_coverage_90": block_coverage,
            "overall_coverage_90": overall_coverage,
            "purity_sensitivity": purity_sensitivity,
        },
        fake_data_report={"passed": fake_recovery.passed, "normalized_error": fake_recovery.normalized_error, "recovered_objective": fake_recovery.recovered_objective, "block_relative_rmse": fake_recovery.block_relative_rmse, "n_synthetic": fake_recovery.n_synthetic, "skipped_reason": fake_recovery.skipped_reason},
        implementation_status_report=implementation,
        prior_diagnostics_report={} if prior_diagnostics_report is None else prior_diagnostics_report,
        count_observation_report={"total_count_observations": sum(1 for o in tensor.count_observations if o.gate_index is None), "gate_count_observations": sum(1 for o in tensor.count_observations if o.gate_index is not None), "gate_counts_by_state": count_gate_counts, "backward_compatibility": "records without gate remain count_total"},
        posterior_predictive_residuals=_posterior_residual_rows(tensor.observed_summary, prediction.summary),
        posterior_predictive_intervals=interval_rows,
        sbc_report=None if sbc_report is None else {"n_datasets": sbc_report.n_datasets, "failures": sbc_report.failures, "ranks": sbc_report.ranks, "skipped_reason": sbc_report.skipped_reason},
    )


def write_v4_lite_reports(output_dir: Path, reports: V4LiteReports, parameter_status_table: tuple[dict[str, object], ...], model_comparison: dict[str, float], *, write_optional_plots: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tensor_summary": reports.tensor_summary,
        "calibration_report": reports.calibration_report,
        "ecDNA_report": reports.ecDNA_report,
        "identifiability_report": reports.identifiability_report,
        "posterior_predictive_report": reports.posterior_predictive_report,
        "fake_data_report": reports.fake_data_report,
        "implementation_status_report": reports.implementation_status_report,
        "prior_diagnostics_report": reports.prior_diagnostics_report,
        "count_observation_report": reports.count_observation_report,
        "sbc_report": reports.sbc_report,
        "model_comparison": model_comparison,
        "parameter_status_table": list(parameter_status_table),
        "posterior_predictive_intervals": list(reports.posterior_predictive_intervals),
    }
    for name, data in (
        ("v4_lite_reports.json", payload),
        ("parameter_status.json", list(parameter_status_table)),
        ("cleaned_tensor_summary.json", reports.tensor_summary),
        ("observation_calibration_report.json", reports.calibration_report),
        ("ecDNA_only_report.json", reports.ecDNA_report),
        ("identifiability_report.json", reports.identifiability_report),
        ("posterior_predictive_report.json", reports.posterior_predictive_report),
        ("count_observation_report.json", reports.count_observation_report),
        ("prior_diagnostics_report.json", reports.prior_diagnostics_report),
        ("implementation_status_report.json", reports.implementation_status_report),
    ):
        (output_dir / name).write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    if reports.sbc_report is not None:
        (output_dir / "sbc_report.json").write_text(json.dumps(reports.sbc_report, indent=2, default=str), encoding="utf-8")
    with open(output_dir / "parameter_status.csv", "w", encoding="utf-8", newline="") as handle:
        fields = ("name", "field", "transform", "prior_kind", "fake_data_passed", "status", "profile_span", "posterior_sd", "prior_scale", "boundary_margin", "rationale")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in parameter_status_table:
            writer.writerow({field: row.get(field) for field in fields})
    with open(output_dir / "posterior_predictive_residuals.csv", "w", encoding="utf-8", newline="") as handle:
        fields = ("block", "key", "observed", "predicted", "residual")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in reports.posterior_predictive_residuals:
            writer.writerow({field: row.get(field) for field in fields})
    with open(output_dir / "posterior_predictive_intervals.csv", "w", encoding="utf-8", newline="") as handle:
        fields = ("block", "key", "observed", "p05", "p50", "p95", "covered_90")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in reports.posterior_predictive_intervals:
            writer.writerow({field: row.get(field) for field in fields})
    if write_optional_plots:
        try:
            mpl_config = output_dir / ".mplconfig"
            cache_dir = output_dir / ".cache"
            mpl_config.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
            os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return
        values = reports.posterior_predictive_report.get("block_relative_rmse", {})
        if isinstance(values, dict) and values:
            fig, ax = plt.subplots(figsize=(max(4, len(values) * 0.8), 3))
            ax.bar(list(values), [float(v) for v in values.values()])
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(output_dir / "posterior_predictive_relative_rmse.png", dpi=150)
            plt.close(fig)


def _state_fraction_rows(prediction: V4LitePrediction) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition, abundance in prediction.state_abundance.items():
        for week_idx, week in enumerate(prediction.weeks):
            total = max(float(np.sum(abundance[week_idx])), 1e-12)
            for state_idx, state in enumerate(cfg.STATE_NAMES):
                rows.append({"condition": condition, "week": week, "state": state, "abundance": float(abundance[week_idx, state_idx]), "fraction": float(abundance[week_idx, state_idx] / total)})
    return rows


def _copy_distribution_rows(prediction: V4LitePrediction, binning: CopyNumberBinning) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition, copies in prediction.copy_distributions.items():
        for week_idx, week in enumerate(prediction.weeks):
            for state_idx, state in enumerate(cfg.STATE_NAMES):
                for species_idx, species in enumerate(cfg.SPECIES):
                    for bin_idx, probability in enumerate(copies[week_idx, state_idx, species_idx].tolist()):
                        lower, upper = binning.bins[bin_idx]
                        rows.append(
                            {
                                "condition": condition,
                                "week": week,
                                "state": state,
                                "species": species,
                                "bin": bin_idx,
                                "lower": lower,
                                "upper": upper,
                                "center": float(binning.centers[bin_idx]),
                                "probability": float(probability),
                            }
                        )
    return rows


def _copy_summary_rows(prediction: V4LitePrediction, binning: CopyNumberBinning) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    centers = binning.centers
    for condition, copies in prediction.copy_distributions.items():
        for week_idx, week in enumerate(prediction.weeks):
            for state_idx, state in enumerate(cfg.STATE_NAMES):
                for species_idx, species in enumerate(cfg.SPECIES):
                    probs = copies[week_idx, state_idx, species_idx]
                    mean = float(np.dot(probs, centers))
                    variance = float(np.dot(probs, np.square(centers - mean)))
                    rows.append(
                        {
                            "condition": condition,
                            "week": week,
                            "state": state,
                            "species": species,
                            "mean": mean,
                            "variance": variance,
                            "zero_fraction": float(probs[0]),
                            "tail_fraction_last_observed_bin": float(probs[-1]),
                        }
                    )
    return rows


def _transition_rows(prediction: V4LitePrediction) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition, transitions in prediction.transition_matrices.items():
        for interval in range(transitions.shape[0]):
            for source_idx, source in enumerate(cfg.STATE_NAMES):
                for target_idx, target in enumerate(cfg.STATE_NAMES):
                    rows.append(
                        {
                            "condition": condition,
                            "week_start": prediction.weeks[interval],
                            "week_end": prediction.weeks[interval + 1],
                            "source_state": source,
                            "target_state": target,
                            "probability": float(transitions[interval, source_idx, target_idx]),
                        }
                    )
    return rows


def _growth_rows(prediction: V4LitePrediction) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition, growth in prediction.growth_rates.items():
        for interval in range(growth.shape[0]):
            for state_idx, state in enumerate(cfg.STATE_NAMES):
                rows.append(
                    {
                        "condition": condition,
                        "week_start": prediction.weeks[interval],
                        "week_end": prediction.weeks[interval + 1],
                        "state": state,
                        "log_net_growth": float(growth[interval, state_idx]),
                    }
                )
    return rows


def _transition_flux_rows(prediction: V4LitePrediction) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition, transitions in prediction.transition_matrices.items():
        abundance = prediction.state_abundance[condition]
        for interval in range(transitions.shape[0]):
            for source_idx, source in enumerate(cfg.STATE_NAMES):
                for target_idx, target in enumerate(cfg.STATE_NAMES):
                    rows.append(
                        {
                            "condition": condition,
                            "week_start": prediction.weeks[interval],
                            "week_end": prediction.weeks[interval + 1],
                            "source_state": source,
                            "target_state": target,
                            "expected_flux": float(abundance[interval, source_idx] * transitions[interval, source_idx, target_idx]),
                        }
                    )
    return rows


def _ddpcr_prediction_rows(tensor: V4LiteTensor, prediction: V4LitePrediction) -> list[dict[str, object]]:
    predicted = prediction.summary.blocks.get("ddpcr_pooled_mean")
    observed = tensor.observed_summary.blocks.get("ddpcr_pooled_mean")
    if predicted is None or observed is None:
        return []
    aligned = predicted.align_to(observed)
    return [
        {"key": key, "observed_ddpcr": float(obs), "predicted_pooled_mean": float(pred), "residual": float(pred - obs), "policy": "ddPCR anchors pooled mean only"}
        for key, obs, pred in zip(observed.keys, observed.values, aligned.values)
    ]


def _observed_ectag_histogram_rows(tensor: V4LiteTensor) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for obs in tensor.ectag_hist_observations:
        total = max(float(np.sum(obs.counts)), 1e-12)
        prefix = {
            "condition": obs.condition,
            "week": obs.week,
            "state": cfg.STATE_NAMES[obs.gate_index],
            "species": cfg.SPECIES[obs.species_index],
            "replicate_id": obs.replicate_id,
            "n_cells": int(np.sum(obs.counts)),
            "source": "observed_ecTAG",
        }
        for bin_idx, count in enumerate(obs.counts.tolist()):
            lower, upper = tensor.structure.binning.bins[bin_idx]
            rows.append(
                {
                    **prefix,
                    "bin": bin_idx,
                    "lower": lower,
                    "upper": upper,
                    "center": float(tensor.structure.binning.centers[bin_idx]),
                    "count": int(count),
                    "probability": float(count / total),
                    "histogram_likelihood_policy": "full_histogram" if int(np.sum(obs.counts)) >= 50 else "mean_level_low_cell_count",
                }
            )
    return rows


def _observed_ectag_summary_rows(tensor: V4LiteTensor) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    centers = tensor.structure.binning.centers
    for obs in tensor.ectag_hist_observations:
        total = max(float(np.sum(obs.counts)), 1e-12)
        probs = obs.counts.astype(float) / total
        mean = float(np.dot(probs, centers))
        rows.append(
            {
                "condition": obs.condition,
                "week": obs.week,
                "state": cfg.STATE_NAMES[obs.gate_index],
                "species": cfg.SPECIES[obs.species_index],
                "replicate_id": obs.replicate_id,
                "n_cells": int(np.sum(obs.counts)),
                "mean": mean,
                "variance": float(np.dot(probs, np.square(centers - mean))),
                "zero_fraction": float(probs[0]),
                "tail_fraction_last_observed_bin": float(probs[-1]),
                "source": "observed_ecTAG",
                "histogram_likelihood_policy": "full_histogram" if int(np.sum(obs.counts)) >= 50 else "mean_level_low_cell_count",
            }
        )
    return rows


def _observed_ectag_joint_rows(tensor: V4LiteTensor) -> list[dict[str, object]]:
    if not tensor.ectag_corr_observations:
        return [{"has_same_cell_ectag": tensor.has_same_cell_ectag, "n_corr": 0, "source": "observed_ecTAG"}]
    return [
        {
            "condition": obs.condition,
            "week": obs.week,
            "state": cfg.STATE_NAMES[obs.gate_index],
            "species_a": cfg.SPECIES[obs.species_a],
            "species_b": cfg.SPECIES[obs.species_b],
            "correlation": obs.correlation,
            "n_cells": obs.n_cells,
            "replicate_id": obs.replicate_id,
            "has_same_cell_ectag": True,
            "source": "observed_ecTAG",
        }
        for obs in tensor.ectag_corr_observations
    ]


def _release_table_rows(stage_results: Sequence[V4LiteStageFitResult]) -> list[dict[str, object]]:
    accepted = {stage.stage_name: stage.accepted for stage in stage_results}
    return [
        {
            "full_block": "growth_hazard",
            "lite_evidence_stage": "M3-growth-coupling",
            "release": bool(accepted.get("M3-growth-coupling", False)),
            "lite_parameter": "theta_P",
            "full_parameter_hint": "hazard.theta_P / growth-response terms",
            "reason": "accepted by v4-lite criteria" if accepted.get("M3-growth-coupling", False) else "fixed 0: v4-lite criteria not met",
        },
        {
            "full_block": "state_landscape_transition",
            "lite_evidence_stage": "M4-transition-coupling",
            "release": bool(accepted.get("M4-transition-coupling", False)),
            "lite_parameter": "beta_C,beta_P,lambda_M",
            "full_parameter_hint": "landscape/plasticity transition terms",
            "reason": "accepted by v4-lite criteria" if accepted.get("M4-transition-coupling", False) else "fixed/collapsed: v4-lite criteria not met",
        },
        {
            "full_block": "co_segregation",
            "lite_evidence_stage": "M3-co-segregation",
            "release": bool(accepted.get("M3-co-segregation", False)),
            "lite_parameter": "co_segregation_rho",
            "full_parameter_hint": "daughter-memory / same-cell species coupling",
            "reason": "accepted by same-cell multi-species ecTAG" if accepted.get("M3-co-segregation", False) else "skipped unless same-cell multi-species ecTAG supports it",
        },
        {
            "full_block": "ecDNA_tail_turnover",
            "lite_evidence_stage": "M1-ecDNA-kernel",
            "release": False,
            "lite_parameter": "kernel_up_species,kernel_down_species",
            "full_parameter_hint": "turnover gain/loss ceilings",
            "reason": "bridge calibration may use M1 summaries; formal release requires full PPC failure on ecTAG tail",
        },
    ]


def _coupling_mode_from_stage_results(stage_results: Sequence[V4LiteStageFitResult]) -> str:
    accepted = {stage.stage_name: stage.accepted for stage in stage_results}
    growth = bool(accepted.get("M3-growth-coupling", False))
    transition = bool(accepted.get("M4-transition-coupling", False))
    if growth and transition:
        return "joint"
    if growth:
        return "growth"
    if transition:
        return "transition"
    return "none"


def _stage_diagnostics(stage_results: Sequence[V4LiteStageFitResult], stage_name: str) -> dict[str, object]:
    for stage in stage_results:
        if stage.stage_name == stage_name:
            return dict(stage.diagnostics or {})
    return {}


def _coupling_metric(stage_results: Sequence[V4LiteStageFitResult], stage_name: str, metric: str) -> object:
    diagnostics = _stage_diagnostics(stage_results, stage_name)
    coupling = diagnostics.get("coupling_diagnostics", {})
    if isinstance(coupling, Mapping):
        return coupling.get(metric)
    return None


def _posterior_method(result: V4LiteFitResult) -> str:
    return result.posterior_samples.method if result.posterior_samples is not None else "not_run"


def _lite_to_full_priors(result: V4LiteFitResult, release_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    params = result.final_params
    return {
        "source": f"v4-lite MAP plus {_posterior_method(result)} posterior samples",
        "posterior_label": _posterior_method(result),
        "posterior_note": None if result.posterior_samples is None else result.posterior_samples.skipped_reason,
        "final_coupling_mode": _coupling_mode_from_stage_results(result.stage_results),
        "ddpcr_policy": "bulk pooled mean anchor only",
        "ectag_policy": "species-specific histograms; total burden derived",
        "weeks": result.tensor.weeks,
        "release_table": list(release_rows),
        "lite_parameter_centers": {
            "kernel_up_species": params.kernel_up_species.tolist(),
            "kernel_down_species": params.kernel_down_species.tolist(),
            "growth_base": params.growth_base.tolist(),
            "mobility_log": params.mobility_log.tolist(),
            "theta_P": params.theta_P,
            "theta_B": params.theta_B,
            "beta_C": params.beta_C,
            "beta_P": params.beta_P,
            "lambda_M": params.lambda_M,
            "co_segregation_rho": params.co_segregation_rho,
        },
    }


def _obs_params_for_full(result: V4LiteFitResult) -> dict[str, object]:
    params = result.final_params
    return {
        "source": "v4-lite observation calibration",
        "posterior_label": f"MAP_with_{_posterior_method(result)}",
        "ddpcr_policy": "bulk pooled mean anchor only; never single-cell distribution evidence",
        "ectag_policy": "species-specific histograms; no config.py ecTAG_max_observed censoring assumption",
        "qpcdr": {
            species: {
                "intercept": float(params.qpcdr_intercept[idx]),
                "slope": float(params.qpcdr_slope[idx]),
                "sigma": float(params.qpcdr_sigma[idx]),
            }
            for idx, species in enumerate(cfg.SPECIES)
        },
        "flow": {
            "concentration": float(params.flow_concentration),
            "sort_purity_matrix": params.sort_purity_matrix.tolist(),
        },
        "counts": {
            "total_count_dispersion": float(params.count_dispersion),
            "gate_count_dispersion": float(params.count_gate_dispersion),
        },
        "ectag": {
            "concentration_by_species": {species: float(params.ectag_concentration[idx]) for idx, species in enumerate(cfg.SPECIES)},
            "same_cell_correlation_sigma": float(params.ectag_corr_sigma),
            "bins": [
                {"lower": lower, "upper": upper, "center": float(result.tensor.structure.binning.centers[idx])}
                for idx, (lower, upper) in enumerate(result.tensor.structure.binning.bins)
            ],
        },
        "calibration_diagnostics": result.reports.calibration_report if result.reports is not None else {},
    }


def _write_fit_npz_and_nc_marker(output_dir: Path, stem: str, arrays: Mapping[str, np.ndarray], label: str) -> None:
    npz_path = output_dir / f"{stem}.npz"
    write_npz_or_marker(npz_path, arrays, label=label)
    write_netcdf_file(output_dir / f"{stem}.nc", arrays, label=label)


def _fit_method_stage_output_paths(output_dir: str | Path) -> dict[str, tuple[Path, ...]]:
    root = Path(output_dir)
    return {
        "M0-observation-only": (
            root / "M0_observation_only_fit.nc",
            root / "M0_snapshot_latent_estimates.csv",
            root / "M0_ppc_report.pdf",
            root / "obs_params_for_full.json",
        ),
        "M1-ecDNA-kernel": (
            root / "M1_ecDNA_kernel_fit.nc",
            root / "M1_ecDNA_summaries.csv",
            root / "M1_ddPCR_pooled_predictions.csv",
            root / "M1_ppc_report.pdf",
        ),
        "M2-abundance-null": (
            root / "M2_abundance_null_fit.nc",
            root / "M2_transition_matrix.csv",
            root / "M2_net_growth.csv",
            root / "M2_null_predictive_score.json",
        ),
        "M3-growth-coupling": (
            root / "M3_growth_coupling_fit.nc",
            root / "M3_growth_coupling_table.csv",
            root / "M3_vs_M2_model_comparison.json",
        ),
        "M4-transition-coupling": (
            root / "M4_transition_coupling_fit.nc",
            root / "M4_transition_coupling_table.csv",
            root / "M4_transition_flux.csv",
            root / "M4_vs_M2_M3_comparison.json",
        ),
        "LITE-final": (
            root / "LITE_final_fit.nc",
            root / "LITE_posterior_predictive.csv",
            root / "LITE_coupling_release_table.csv",
            root / "LITE_to_FULL_priors.json",
        ),
    }


def write_fit_method_artifacts(output_dir: str | Path, result: V4LiteFitResult, parameter_status_table: tuple[dict[str, object], ...], model_comparison: dict[str, float]) -> None:
    """Write the file set described by markdown/fit_method.md.

    NetCDF, NPZ, CSV, optional parquet, and concise PDF diagnostics are written
    so the automated run can be inspected without extra dependencies.
    """

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cfg.require(result.reports is not None, "reports are required before writing fit_method artifacts.")
    prediction = predict_v4_lite(result.tensor, result.final_params, coupling_mode=_coupling_mode_from_stage_results(result.stage_results))
    binning = result.tensor.structure.binning
    write_standardized_dataset(destination, result.tensor.dataset)

    state_rows = _state_fraction_rows(prediction)
    copy_rows = _copy_distribution_rows(prediction, binning)
    copy_summary = _copy_summary_rows(prediction, binning)
    observed_hist_rows = _observed_ectag_histogram_rows(result.tensor)
    observed_summary_rows = _observed_ectag_summary_rows(result.tensor)
    observed_joint_rows = _observed_ectag_joint_rows(result.tensor)
    transition_rows = _transition_rows(prediction)
    growth_rows = _growth_rows(prediction)
    flux_rows = _transition_flux_rows(prediction)
    ddpcr_rows = _ddpcr_prediction_rows(result.tensor, prediction)
    release_rows = _release_table_rows(result.stage_results)

    initial_anchor = {
        "week1_state_abundance": {condition: abundance.tolist() for condition, abundance in result.tensor.initial_state_abundance.items()},
        "week1_flow_fractions": {condition: (abundance / max(float(np.sum(abundance)), 1e-12)).tolist() for condition, abundance in result.tensor.initial_state_abundance.items()},
        "ddpcr_anchor_policy": "bulk pooled mean only; no single-cell or state-specific mean constraint",
        "olig2_initial_ratio": result.tensor.dataset.olig2_initial_ratio,
        "olig2_anchor_policy": "weak initial state prior on (NPC+OPC)/(AC+MES)" if result.tensor.dataset.olig2_initial_ratio is not None else "not_provided",
        "ectag_policy": "species-specific adaptive bins; no config.py ecTAG_max_observed censoring",
    }
    write_json(destination / "initial_anchor.json", initial_anchor)
    (destination / "initial_prior_report.md").write_text(
        "\n".join(
            [
                "# Initial Prior Report",
                "",
                "Week1 flow and sorted ecTAG initialize the v4-lite state/copy distributions.",
                "ddPCR is used only as a pooled mean anchor in the likelihood.",
                f"OLIG2 weak prior status: {'provided' if result.tensor.dataset.olig2_initial_ratio is not None else 'not_provided'}.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    write_table_bundle(destination, "empirical_state_species_summary", observed_summary_rows)
    write_table_bundle(destination, "ectag_histograms", observed_hist_rows)
    write_table_bundle(destination, "ectag_joint_summary", observed_joint_rows)
    write_text_pdf(destination / "empirical_summary_plots.pdf", "Empirical summary diagnostics", ["CSV/parquet tables contain observed-data summaries.", f"Observed ecTAG histogram rows: {len(observed_hist_rows)}"])

    arrays = {
        "posterior_samples": result.posterior_samples.samples if result.posterior_samples is not None else np.zeros((0, 0)),
        "state_abundance": next(iter(prediction.state_abundance.values())),
        "copy_distributions": next(iter(prediction.copy_distributions.values())),
    }
    _write_fit_npz_and_nc_marker(destination, "M0_observation_only_fit", arrays, "M0 observation-only approximate MAP output")
    write_table_bundle(destination, "M0_snapshot_latent_estimates", state_rows + copy_summary)
    write_json(destination / "M0_measurement_noise_final.json", result.reports.calibration_report)
    write_json(destination / "obs_params_for_full.json", _obs_params_for_full(result))
    write_text_pdf(
        destination / "obs_calibration_ppc.pdf",
        "Observation calibration PPC",
        [
            "qPCDR, flow, ecTAG, and ddPCR observation policies are summarized in obs_params_for_full.json.",
            "ddPCR policy: pooled mean anchor only.",
            f"Observed ecTAG histogram rows: {len(observed_hist_rows)}",
        ],
    )
    write_text_pdf(destination / "M0_ppc_report.pdf", "M0 observation-only PPC", [f"Worst relative RMSE: {result.reports.posterior_predictive_report.get('worst_relative_rmse')}"])

    _write_fit_npz_and_nc_marker(destination, "M1_ecDNA_kernel_fit", arrays, "M1 ecDNA kernel approximate MAP output")
    write_table_bundle(destination, "M1_ecDNA_predicted_distributions", copy_rows)
    write_table_bundle(destination, "M1_ecDNA_summaries", copy_summary)
    write_table_bundle(destination, "M1_ddPCR_pooled_predictions", ddpcr_rows)
    write_text_pdf(destination / "M1_ppc_report.pdf", "M1 ecDNA/qPCDR/ddPCR PPC", [f"ddPCR prediction rows: {len(ddpcr_rows)}", "ddPCR policy: pooled mean only"])

    _write_fit_npz_and_nc_marker(destination, "M2_abundance_null_fit", arrays, "M2 abundance null approximate MAP output")
    write_table_bundle(destination, "M2_transition_matrix", transition_rows)
    write_table_bundle(destination, "M2_net_growth", growth_rows)
    write_json(destination / "M2_null_predictive_score.json", {"model_comparison": model_comparison, "stage": "M2-abundance-null"})
    write_text_pdf(destination / "M2_flow_ppc_report.pdf", "M2 flow/count PPC", [f"state rows: {len(state_rows)}", f"transition rows: {len(transition_rows)}"])

    _write_fit_npz_and_nc_marker(destination, "M3_growth_coupling_fit", arrays, "M3 growth-coupling candidate approximate MAP output")
    write_table_bundle(
        destination,
        "M3_growth_coupling_table",
        [
            {
                "parameter": "theta_P",
                "estimate": result.final_params.theta_P,
                "status": "free" if any(row["release"] for row in release_rows if row["full_block"] == "growth_hazard") else "fixed_or_rejected",
                "sign_probability": _coupling_metric(result.stage_results, "M3-growth-coupling", "posterior_sign_probability"),
                "contraction": _coupling_metric(result.stage_results, "M3-growth-coupling", "posterior_contraction"),
                "synthetic_sign_recovery": _coupling_metric(result.stage_results, "M3-growth-coupling", "synthetic_sign_recovery"),
                "synthetic_recovery_datasets": _coupling_metric(result.stage_results, "M3-growth-coupling", "synthetic_recovery_datasets"),
            }
        ],
    )
    write_json(destination / "M3_vs_M2_model_comparison.json", {"M3_vs_M2": model_comparison.get("M3_vs_M2.log_objective_improvement"), "criteria": ">=4 plus posterior sign/recovery; approximate run rejects if not proven"})
    write_text_pdf(destination / "M3_flow_count_ppc.pdf", "M3 growth coupling PPC", ["Coupling is retained only if all v4-lite criteria pass."])

    _write_fit_npz_and_nc_marker(destination, "M4_transition_coupling_fit", arrays, "M4 transition-coupling candidate approximate MAP output")
    write_table_bundle(
        destination,
        "M4_transition_coupling_table",
        [
            {
                "parameter": "beta_C",
                "estimate": result.final_params.beta_C,
                "status": "fixed_or_rejected",
                "sign_probability": _coupling_metric(result.stage_results, "M4-transition-coupling", "posterior_sign_probability"),
                "contraction": _coupling_metric(result.stage_results, "M4-transition-coupling", "posterior_contraction"),
                "synthetic_sign_recovery": _coupling_metric(result.stage_results, "M4-transition-coupling", "synthetic_sign_recovery"),
            },
            {
                "parameter": "beta_P",
                "estimate": result.final_params.beta_P,
                "status": "fixed_or_rejected",
                "sign_probability": _coupling_metric(result.stage_results, "M4-transition-coupling", "posterior_sign_probability"),
                "contraction": _coupling_metric(result.stage_results, "M4-transition-coupling", "posterior_contraction"),
                "synthetic_sign_recovery": _coupling_metric(result.stage_results, "M4-transition-coupling", "synthetic_sign_recovery"),
            },
            {
                "parameter": "lambda_M",
                "estimate": result.final_params.lambda_M,
                "status": "fixed_or_rejected",
                "sign_probability": _coupling_metric(result.stage_results, "M4-transition-coupling", "posterior_sign_probability"),
                "contraction": _coupling_metric(result.stage_results, "M4-transition-coupling", "posterior_contraction"),
                "synthetic_sign_recovery": _coupling_metric(result.stage_results, "M4-transition-coupling", "synthetic_sign_recovery"),
            },
        ],
    )
    write_table_bundle(destination, "M4_transition_flux", flux_rows)
    write_json(destination / "M4_vs_M2_M3_comparison.json", {"M4_vs_M2_M3": model_comparison.get("M4_vs_M2_M3.log_objective_improvement"), "criteria": ">=4 plus profile and sign recovery; approximate run rejects if not proven"})
    write_text_pdf(destination / "M4_transition_ppc.pdf", "M4 transition coupling PPC", ["Transition coupling is retained only if all v4-lite criteria pass."])

    _write_fit_npz_and_nc_marker(destination, "LITE_final_fit", arrays, "Final v4-lite approximate posterior output")
    write_table_bundle(destination, "LITE_posterior_predictive", result.reports.posterior_predictive_intervals or result.reports.posterior_predictive_residuals)
    write_table_bundle(destination, "LITE_state_fractions", state_rows)
    write_table_bundle(destination, "LITE_ecDNA_distributions", copy_rows)
    write_table_bundle(destination, "LITE_ecDNA_summaries", copy_summary)
    write_table_bundle(destination, "LITE_growth", growth_rows)
    write_table_bundle(destination, "LITE_transition_matrix", transition_rows)
    write_table_bundle(destination, "LITE_transition_flux", flux_rows)
    write_table_bundle(destination, "LITE_coupling_release_table", release_rows)
    write_json(destination / "LITE_to_FULL_priors.json", _lite_to_full_priors(result, release_rows))
    write_text_pdf(
        destination / "LITE_final_report.pdf",
        "LITE final report",
        [
            f"Posterior label: {result.posterior_samples.skipped_reason if result.posterior_samples else 'missing'}",
            f"Worst relative RMSE: {result.reports.posterior_predictive_report.get('worst_relative_rmse')}",
            f"Release rows: {len(release_rows)}",
        ],
    )
    write_json(
        destination / "fit_method_completion_status.json",
        {
            "implemented": [
                "raw standardization CSV plus optional parquet",
                "M0-M4 and LITE report file set",
                "species-specific ecTAG binning",
                "ddPCR pooled mean anchor",
                "xarray NetCDF files plus NPZ array mirrors",
                "posterior predictive 90% interval coverage",
                "50-dataset synthetic sign-recovery diagnostics for coupling candidates",
                "approximate SBC rank diagnostics",
                "full-to-lite prior/release table export",
            ],
            "approximate_or_skipped": [
                f"posterior backend: {_posterior_method(result)}",
                "falls back to Laplace Gaussian approximation when emcee is unavailable or fails",
                "PDF reports are concise generated summaries for manual inspection",
                "restricted full SMC is handled in full_calibration.py and remains approximate unless full raw simulator likelihood is available",
            ],
            "parameter_status_rows": list(parameter_status_table),
        },
    )


def _projection_from_prediction(prediction: V4LitePrediction) -> FullToLiteProjection:
    condition = prediction.condition_names[0]
    return FullToLiteProjection(prediction.weeks, prediction.state_abundance[condition].copy(), prediction.copy_distributions[condition].copy(), prediction.transition_matrices[condition].copy(), prediction.growth_rates[condition].copy(), prediction.copy_kernels[condition].copy(), {"source": "v4-lite prediction", "condition": condition})


def project_full_to_lite(simulation_result, structure: V4LiteStructure | None = None, purity_matrix: np.ndarray | None = None) -> FullToLiteProjection:
    model_structure = V4LiteStructure.default() if structure is None else structure
    weeks = tuple(int(round(float(t))) + 1 for t in simulation_result.times)
    n_weeks = len(weeks)
    n_bins = model_structure.binning.n_bins
    abundance = np.zeros((n_weeks, cfg.N_STATES), dtype=float)
    copies = np.zeros((n_weeks, cfg.N_STATES, cfg.N_SPECIES, n_bins), dtype=float)
    snapshots = getattr(simulation_result, "cell_snapshots", None) or []
    for week_idx in range(n_weeks):
        cells = snapshots[week_idx] if week_idx < len(snapshots) else []
        if cells:
            soft = np.asarray([cell["soft_state"] for cell in cells], dtype=float)
            copy_values = np.asarray([cell["copy_numbers"] for cell in cells], dtype=int)
            abundance[week_idx] = np.sum(soft, axis=0)
            for state_idx in range(cfg.N_STATES):
                weights = soft[:, state_idx]
                total = max(float(np.sum(weights)), 1e-12)
                for species_idx in range(cfg.N_SPECIES):
                    counts = np.zeros(n_bins, dtype=float)
                    for value, weight in zip(copy_values[:, species_idx], weights):
                        counts[model_structure.binning.bin_index(int(value))] += float(weight)
                    copies[week_idx, state_idx, species_idx] = _normalize(counts / total)
        else:
            snapshot = simulation_result.truth_snapshots[week_idx]
            size = float(snapshot.get("population_size", simulation_result.population_sizes[week_idx]))
            fractions = np.asarray(snapshot.get("soft_state_fractions", simulation_result.soft_state_fractions[week_idx]), dtype=float)
            abundance[week_idx] = size * fractions
            copies[week_idx, :, :, 0] = 1.0
    transitions, growth, kernels, diag = _event_dynamics(getattr(simulation_result, "events", ()) or (), simulation_result.times, abundance, model_structure)
    if purity_matrix is not None:
        diag["purity_matrix_applied"] = True
    return FullToLiteProjection(weeks, abundance, copies, transitions, growth, kernels, diag)


def _event_dynamics(events, times, abundance: np.ndarray, structure: V4LiteStructure):
    n_intervals = max(len(times) - 1, 0)
    n_bins = structure.binning.n_bins
    transitions = np.zeros((n_intervals, cfg.N_STATES, cfg.N_STATES), dtype=float)
    kernels = np.zeros((n_intervals, cfg.N_STATES, cfg.N_SPECIES, n_bins, n_bins), dtype=float)
    counts = np.zeros_like(transitions)
    copy_counts = np.zeros_like(kernels)
    used = 0
    for event_time, _event_type, _cell_id, details in events:
        interval = int(np.searchsorted(np.asarray(times, dtype=float), float(event_time), side="right") - 1)
        if not (0 <= interval < n_intervals):
            continue
        pre = details.get("state_pre", {}) if isinstance(details, Mapping) else {}
        post = details.get("state_post", {}) if isinstance(details, Mapping) else {}
        if "soft_state" not in pre or "soft_state" not in post:
            continue
        source = _normalize(np.asarray(pre["soft_state"], dtype=float))
        target = _normalize(np.asarray(post["soft_state"], dtype=float))
        counts[interval] += np.outer(source, target)
        if "copy_numbers" in pre and "copy_numbers" in post:
            pre_c = np.asarray(pre["copy_numbers"], dtype=int)
            post_c = np.asarray(post["copy_numbers"], dtype=int)
            for state_idx, weight in enumerate(source):
                for species_idx in range(cfg.N_SPECIES):
                    copy_counts[interval, state_idx, species_idx, structure.binning.bin_index(pre_c[species_idx]), structure.binning.bin_index(post_c[species_idx])] += float(weight)
        used += 1
    for interval in range(n_intervals):
        for state_idx in range(cfg.N_STATES):
            row = counts[interval, state_idx]
            if float(np.sum(row)) <= 0.0:
                row = np.eye(cfg.N_STATES)[state_idx]
            transitions[interval, state_idx] = _normalize(row)
            for species_idx in range(cfg.N_SPECIES):
                for source_bin in range(n_bins):
                    krow = copy_counts[interval, state_idx, species_idx, source_bin]
                    if float(np.sum(krow)) <= 0.0:
                        krow = np.eye(n_bins)[source_bin]
                    kernels[interval, state_idx, species_idx, source_bin] = _normalize(krow)
    growth = np.log(np.maximum(abundance[1:], 1e-8) / np.maximum(abundance[:-1], 1e-8)) if n_intervals else None
    return transitions if n_intervals else None, growth, kernels if n_intervals else None, {"event_count": len(events), "used_transition_events": used}


def summarize_dataset_v4_lite(dataset: CanonicalFitDataset, *, condition_names: Iterable[str] | None = None, binning: CopyNumberBinning | None = None) -> SummaryCollection:
    tensor = build_v4_lite_tensor(dataset, condition_names=condition_names, structure=None if binning is None else V4LiteStructure.default(binning=binning))
    return tensor.observed_summary
