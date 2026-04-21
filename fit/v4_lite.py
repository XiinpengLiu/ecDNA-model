"""
Week-level v4-lite fitting path.

The default fitting path intentionally avoids the full agent-based simulator.
It fits week-level state abundance, state-specific binned ecDNA distributions,
and the sorted-gate observation model described in ``markdown/fit_method.md``.
"""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.special import gammaln

import config as cfg
from fit.data import CanonicalFitDataset, EcTAGRecord, FlowRecord, QPCDRRecord, WEEK1
from fit.summary_types import SummaryCollection


INVALID_OBJECTIVE = 1e18
V4LiteModelVersion = {"M0", "M1", "M2", "M3"}
V4LiteDynamicsMode = {"joint", "ecDNA_only", "state_only"}
DEFAULT_COPY_BINS = ((0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, None))
DEFAULT_COPY_BIN_CENTERS = (0.0, 1.0, 2.5, 5.5, 11.5, 24.0)
DEFAULT_DIRECTED_EDGES = ((cfg.NPC, cfg.OPC), (cfg.OPC, cfg.NPC), (cfg.OPC, cfg.AC), (cfg.AC, cfg.OPC), (cfg.AC, cfg.MES), (cfg.MES, cfg.AC))
OPTIONAL_DIRECTED_EDGES = ((cfg.NPC, cfg.AC), (cfg.AC, cfg.NPC))


def _safe_log1p(value: float | np.ndarray) -> float | np.ndarray:
    return np.log1p(np.clip(value, 0.0, None))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=float) - float(np.max(values))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    cfg.require(np.isfinite(total) and total > 0.0, "Softmax weights must have positive finite mass.")
    return weights / total


def _normalize_simplex(values: np.ndarray, *, floor: float = 1e-12) -> np.ndarray:
    flat = np.clip(np.asarray(values, dtype=float), floor, None)
    total = float(np.sum(flat))
    cfg.require(np.isfinite(total) and total > 0.0, "Simplex values must have positive finite mass.")
    return flat / total


def _student_t_logpdf(value: float, location: float, scale: float, df: float) -> float:
    scale = max(float(scale), 1e-8)
    z = (float(value) - float(location)) / scale
    return float(
        gammaln((df + 1.0) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * np.log(df * np.pi)
        - np.log(scale)
        - ((df + 1.0) / 2.0) * np.log1p((z * z) / df)
    )


def _multinomial_logpmf(counts: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(counts, dtype=float)
    p = _normalize_simplex(probabilities)
    n = float(np.sum(y))
    return float(gammaln(n + 1.0) - np.sum(gammaln(y + 1.0)) + np.dot(y, np.log(np.clip(p, 1e-12, 1.0))))


def _dirichlet_multinomial_logpmf(counts: np.ndarray, probabilities: np.ndarray, concentration: float) -> float:
    y = np.asarray(counts, dtype=float)
    p = _normalize_simplex(probabilities)
    alpha0 = max(float(concentration), 1e-6)
    alpha = np.clip(alpha0 * p, 1e-8, None)
    n = float(np.sum(y))
    return float(
        gammaln(n + 1.0)
        - np.sum(gammaln(y + 1.0))
        + gammaln(alpha0)
        - gammaln(n + alpha0)
        + np.sum(gammaln(y + alpha) - gammaln(alpha))
    )


def _negative_binomial_logpmf(value: float, mean: float, dispersion: float) -> float:
    y = max(float(value), 0.0)
    mu = max(float(mean), 1e-8)
    r = max(float(dispersion), 1e-6)
    p = r / (r + mu)
    return float(gammaln(y + r) - gammaln(r) - gammaln(y + 1.0) + r * np.log(p) + y * np.log1p(-p))


@dataclass(frozen=True)
class CopyNumberBinning:
    bins: tuple[tuple[int, int | None], ...] = DEFAULT_COPY_BINS
    centers: np.ndarray = field(default_factory=lambda: np.asarray(DEFAULT_COPY_BIN_CENTERS, dtype=float))

    def __post_init__(self) -> None:
        cfg.require(len(self.bins) == int(self.centers.size), "Copy-number bin centers must match bins.")
        cfg.require(self.bins[0][0] == 0, "Copy-number bins must start at zero.")

    @property
    def n_bins(self) -> int:
        return len(self.bins)

    def bin_index(self, value: int | float) -> int:
        integer_value = int(max(0, round(float(value))))
        for index, (lower, upper) in enumerate(self.bins):
            if upper is None and integer_value >= lower:
                return index
            if upper is not None and lower <= integer_value <= upper:
                return index
        return self.n_bins - 1

    def counts(self, values: Iterable[int | float]) -> np.ndarray:
        result = np.zeros(self.n_bins, dtype=int)
        for value in values:
            result[self.bin_index(value)] += 1
        return result

    def probabilities(self, values: Iterable[int | float], *, epsilon: float = 0.0) -> np.ndarray:
        counts = self.counts(values).astype(float)
        if epsilon > 0.0:
            counts += float(epsilon)
        if float(np.sum(counts)) <= 0.0:
            counts[0] = 1.0
        return counts / float(np.sum(counts))

    def mean(self, probabilities: np.ndarray) -> float:
        return float(np.dot(np.asarray(probabilities, dtype=float).reshape(self.n_bins), self.centers))

    def tail_probability(self, probabilities: np.ndarray, threshold: int) -> float:
        probs = np.asarray(probabilities, dtype=float).reshape(self.n_bins)
        total = 0.0
        for index, (lower, upper) in enumerate(self.bins):
            if upper is None or upper >= threshold:
                if lower >= threshold or (upper is not None and lower < threshold <= upper):
                    total += float(probs[index])
        return total


@dataclass(frozen=True)
class FlowObservation:
    condition: str
    week: int
    counts: np.ndarray
    replicate_id: str

    @property
    def total(self) -> float:
        return float(np.sum(self.counts))


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
class V4LiteStructure:
    transition_edges: tuple[tuple[int, int], ...]
    binning: CopyNumberBinning = field(default_factory=CopyNumberBinning)
    qpcdr_batches: tuple[str, ...] = ("default",)

    @classmethod
    def default(
        cls,
        *,
        include_optional_edge: bool = False,
        qpcdr_batches: Iterable[str] = ("default",),
    ) -> "V4LiteStructure":
        edges = tuple(DEFAULT_DIRECTED_EDGES + (OPTIONAL_DIRECTED_EDGES if include_optional_edge else ()))
        batches = tuple(qpcdr_batches) or ("default",)
        return cls(transition_edges=edges, qpcdr_batches=tuple(sorted(batches)))

    def with_qpcdr_batches(self, batches: Iterable[str]) -> "V4LiteStructure":
        batch_tuple = tuple(sorted(set(batches))) or ("default",)
        return V4LiteStructure(transition_edges=self.transition_edges, binning=self.binning, qpcdr_batches=batch_tuple)

    @property
    def n_edges(self) -> int:
        return len(self.transition_edges)

    @property
    def undirected_edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted({tuple(sorted(edge)) for edge in self.transition_edges}))

    def mobility_index(self, source: int, target: int) -> int:
        pair = tuple(sorted((int(source), int(target))))
        return self.undirected_edges.index(pair)

    @property
    def n_mobility_edges(self) -> int:
        return len(self.undirected_edges)

    @property
    def n_qpcdr_batches(self) -> int:
        return len(self.qpcdr_batches)


@dataclass
class V4LiteParameters:
    qpcdr_intercept: np.ndarray = field(default_factory=lambda: np.zeros(cfg.N_SPECIES, dtype=float))
    qpcdr_slope: np.ndarray = field(default_factory=lambda: np.ones(cfg.N_SPECIES, dtype=float))
    qpcdr_sigma: np.ndarray = field(default_factory=lambda: np.full(cfg.N_SPECIES, 0.25, dtype=float))
    qpcdr_batch_offsets: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    qpcdr_df: float = 4.0
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
    kernel_down_C_target: float = 0.0
    kernel_down_P_target: float = 0.0
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
    def default(
        cls,
        structure: V4LiteStructure | None = None,
        *,
        purity_matrix: np.ndarray | None = None,
        qpcdr_calibration: Mapping[str, Mapping[str, float]] | None = None,
    ) -> "V4LiteParameters":
        model_structure = V4LiteStructure.default() if structure is None else structure
        params = cls()
        params.qpcdr_batch_offsets = np.zeros(model_structure.n_qpcdr_batches, dtype=float)
        params.alpha_state = params.alpha_state - float(np.mean(params.alpha_state))
        mobility_values: list[float] = []
        for source, target in model_structure.undirected_edges:
            forward = cfg.DEFAULT_MODEL_PARAMETERS.generator.base_edges.get((source, target), 0.08)
            backward = cfg.DEFAULT_MODEL_PARAMETERS.generator.base_edges.get((target, source), forward)
            mobility_values.append(np.log(max(float(0.5 * (forward + backward)), 1e-4)))
        params.mobility_log = np.asarray(mobility_values, dtype=float)
        if purity_matrix is not None:
            params.sort_purity_matrix = _normalize_purity_matrix(purity_matrix)
        if qpcdr_calibration:
            for species_name, calibration in qpcdr_calibration.items():
                if species_name not in cfg.SPECIES_INDEX:
                    continue
                species_index = cfg.SPECIES_INDEX[species_name]
                if "intercept" in calibration:
                    params.qpcdr_intercept[species_index] = float(calibration["intercept"])
                if "slope" in calibration:
                    params.qpcdr_slope[species_index] = max(float(calibration["slope"]), 1e-8)
                if "sigma" in calibration:
                    params.qpcdr_sigma[species_index] = max(float(calibration["sigma"]), 1e-8)
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
    observed_summary: SummaryCollection
    has_total_counts: bool
    has_same_cell_ectag: bool
    burden_star: float

    @property
    def week_to_index(self) -> dict[int, int]:
        return {week: index for index, week in enumerate(self.weeks)}

    @property
    def batch_to_index(self) -> dict[str, int]:
        return {batch: index for index, batch in enumerate(self.structure.qpcdr_batches)}


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


def _normalize_purity_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    cfg.require(matrix.shape == (cfg.N_STATES, cfg.N_STATES), "purity matrix has invalid shape.")
    cfg.require(np.all(np.isfinite(matrix)), "purity matrix must be finite.")
    cfg.require(np.all(matrix >= 0.0), "purity matrix must be non-negative.")
    column_sums = np.sum(matrix, axis=0)
    cfg.require(np.all(column_sums > 0.0), "purity matrix columns must have positive mass.")
    return matrix / column_sums


def _flow_record_count(record: FlowRecord) -> float:
    if record.count is not None:
        return float(record.count)
    if record.fraction is not None and record.total_events is not None:
        return float(record.fraction * record.total_events)
    if record.fraction is not None:
        return float(record.fraction * 1000.0)
    return 0.0


def _week1_total_count(dataset: CanonicalFitDataset, condition_name: str) -> float:
    count_values = [
        float(record.value)
        for record in dataset.counts
        if record.condition == condition_name and record.week == WEEK1 and record.gate is None
    ]
    if count_values:
        return max(float(np.mean(count_values)), 1e-8)
    flow_rows = [record for record in dataset.flow if record.condition == condition_name and record.week == WEEK1]
    counted = [_flow_record_count(record) for record in flow_rows]
    total = float(np.sum(counted))
    return max(total, 1.0) if total > 0.0 else 1.0


def build_v4_lite_tensor(
    dataset: CanonicalFitDataset,
    *,
    condition_names: Iterable[str] | None = None,
    structure: V4LiteStructure | None = None,
) -> V4LiteTensor:
    selected_conditions = tuple(dataset.condition_names() if condition_names is None else tuple(condition_names))
    cfg.require(bool(selected_conditions), "At least one condition is required for v4-lite fitting.")
    qpcdr_batches = dataset.qpcdr_batches()
    model_structure = V4LiteStructure.default(qpcdr_batches=qpcdr_batches) if structure is None else structure.with_qpcdr_batches(qpcdr_batches)
    max_week = max(dataset.dynamic_weeks())
    weeks = tuple(range(WEEK1, int(max_week) + 1))

    initial_state_abundance: dict[str, np.ndarray] = {}
    initial_copy_distributions: dict[str, np.ndarray] = {}
    exposure_C: dict[str, np.ndarray] = {}
    exposure_P: dict[str, np.ndarray] = {}
    burden_values: list[float] = []
    for condition_name in selected_conditions:
        cfg.require(condition_name in dataset.conditions, f"Unknown condition {condition_name}.")
        init_condition = dataset.resolve_initialization_condition(condition_name)
        initialization = dataset.build_empirical_initialization(condition_name)
        cfg.require(initialization.empirical_flow_fractions is not None, "Empirical week1 flow fractions are required.")
        cfg.require(initialization.empirical_sorted_copy_distributions is not None, "Empirical week1 copy distributions are required.")
        total_count = _week1_total_count(dataset, init_condition)
        initial_state_abundance[condition_name] = total_count * np.asarray(initialization.empirical_flow_fractions, dtype=float)
        p0 = np.zeros((cfg.N_STATES, cfg.N_SPECIES, model_structure.binning.n_bins), dtype=float)
        for state_index, state_name in enumerate(cfg.STATE_NAMES):
            matrix = np.asarray(initialization.empirical_sorted_copy_distributions[state_name], dtype=int)
            for species_index in range(cfg.N_SPECIES):
                p0[state_index, species_index, :] = model_structure.binning.probabilities(matrix[:, species_index], epsilon=1e-3)
                burden_values.append(float(np.dot(p0[state_index, species_index, :], np.log1p(model_structure.binning.centers))))
        initial_copy_distributions[condition_name] = p0
        schedules = dataset.conditions[condition_name].build_input_schedules()
        exposure_C[condition_name] = np.asarray([float(schedules["u_C"](float(week - WEEK1) + 0.5)) for week in weeks[:-1]], dtype=float)
        exposure_P[condition_name] = np.asarray([float(schedules["u_P"](float(week - WEEK1) + 0.5)) for week in weeks[:-1]], dtype=float)

    flow_observations = _build_flow_observations(dataset, selected_conditions)
    count_observations = tuple(
        CountObservation(
            record.condition,
            record.week,
            float(record.value),
            record.replicate_id or "__aggregate__",
            None if record.gate is None else cfg.STATE_INDEX[record.gate],
        )
        for record in dataset.counts
        if record.condition in selected_conditions and record.week > WEEK1
    )
    batch_to_index = {batch: index for index, batch in enumerate(model_structure.qpcdr_batches)}
    qpcdr_observations = tuple(
        QPCDRObservation(
            record.condition,
            record.week,
            cfg.STATE_INDEX[record.state],
            cfg.SPECIES_INDEX[record.species],
            float(record.value),
            batch_to_index[record.batch],
            record.replicate_id or "__aggregate__",
        )
        for record in dataset.qpcdr
        if record.condition in selected_conditions and record.week > WEEK1
    )
    ectag_hist_observations, ectag_corr_observations = _build_ectag_observations(dataset, selected_conditions, model_structure.binning)
    observed_summary = _observed_summary_from_observations(
        flow_observations,
        count_observations,
        qpcdr_observations,
        ectag_hist_observations,
        ectag_corr_observations,
        model_structure,
        dataset,
    )
    return V4LiteTensor(
        dataset=dataset,
        structure=model_structure,
        condition_names=selected_conditions,
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
        observed_summary=observed_summary,
        has_total_counts=any(observation.gate_index is None for observation in count_observations),
        has_same_cell_ectag=bool(ectag_corr_observations),
        burden_star=float(np.median(burden_values)) if burden_values else 1.0,
    )


def _build_flow_observations(dataset: CanonicalFitDataset, selected_conditions: tuple[str, ...]) -> tuple[FlowObservation, ...]:
    grouped: dict[tuple[str, int, str], np.ndarray] = {}
    for record in dataset.flow:
        if record.condition not in selected_conditions or record.week <= WEEK1:
            continue
        key = (record.condition, record.week, record.replicate_id or "__aggregate__")
        grouped.setdefault(key, np.zeros(cfg.N_STATES, dtype=float))
        grouped[key][cfg.STATE_INDEX[record.state]] += _flow_record_count(record)
    observations: list[FlowObservation] = []
    for (condition_name, week, replicate_id), counts in sorted(grouped.items()):
        if float(np.sum(counts)) > 0.0:
            observations.append(FlowObservation(condition_name, week, counts.astype(int), replicate_id))
    return tuple(observations)


def _ectag_cell_key(record: EcTAGRecord) -> str:
    replicate_token = "" if record.replicate_id is None else f"{record.replicate_id}|"
    return f"{replicate_token}{record.cell_id}"


def _build_ectag_observations(
    dataset: CanonicalFitDataset,
    selected_conditions: tuple[str, ...],
    binning: CopyNumberBinning,
) -> tuple[tuple[EcTAGHistogramObservation, ...], tuple[EcTAGCorrelationObservation, ...]]:
    hist_groups: dict[tuple[str, int, str, str, str], list[int]] = {}
    cell_groups: dict[tuple[str, int, str, str], dict[str, dict[str, int]]] = {}
    for record in dataset.ectag:
        if record.condition not in selected_conditions or record.week <= WEEK1:
            continue
        replicate_id = record.replicate_id or "__aggregate__"
        hist_groups.setdefault((record.condition, record.week, record.state, record.species, replicate_id), []).append(int(record.value))
        cell_key = _ectag_cell_key(record)
        cell_groups.setdefault((record.condition, record.week, record.state, replicate_id), {}).setdefault(cell_key, {})[record.species] = int(record.value)

    hist_observations: list[EcTAGHistogramObservation] = []
    for (condition_name, week, state_name, species_name, replicate_id), values in sorted(hist_groups.items()):
        counts = binning.counts(values)
        if int(np.sum(counts)) > 0:
            hist_observations.append(
                EcTAGHistogramObservation(condition_name, week, cfg.STATE_INDEX[state_name], cfg.SPECIES_INDEX[species_name], counts, replicate_id)
            )

    corr_observations: list[EcTAGCorrelationObservation] = []
    for (condition_name, week, state_name, replicate_id), cell_map in sorted(cell_groups.items()):
        rows = [[species_map[species_name] for species_name in cfg.SPECIES] for species_map in cell_map.values() if set(species_map) == set(cfg.SPECIES)]
        if len(rows) < 2:
            continue
        matrix = np.asarray(rows, dtype=float)
        if np.any(np.std(matrix, axis=0) == 0.0):
            corr = np.zeros((cfg.N_SPECIES, cfg.N_SPECIES), dtype=float)
        else:
            corr = np.nan_to_num(np.corrcoef(matrix, rowvar=False), nan=0.0, posinf=0.0, neginf=0.0)
        for first in range(cfg.N_SPECIES):
            for second in range(first + 1, cfg.N_SPECIES):
                corr_observations.append(
                    EcTAGCorrelationObservation(condition_name, week, cfg.STATE_INDEX[state_name], first, second, float(corr[first, second]), len(rows), replicate_id)
                )
    return tuple(hist_observations), tuple(corr_observations)


def _observed_summary_from_observations(
    flow: tuple[FlowObservation, ...],
    counts: tuple[CountObservation, ...],
    qpcdr: tuple[QPCDRObservation, ...],
    ectag_hist: tuple[EcTAGHistogramObservation, ...],
    ectag_corr: tuple[EcTAGCorrelationObservation, ...],
    structure: V4LiteStructure,
    dataset: CanonicalFitDataset,
) -> SummaryCollection:
    block_maps = _empty_v4_lite_block_maps()
    for observation in flow:
        fractions = observation.counts / max(float(np.sum(observation.counts)), 1e-12)
        for state_index, state_name in enumerate(cfg.STATE_NAMES):
            key = f"{observation.condition}|week{observation.week}|state={state_name}|rep={observation.replicate_id}"
            block_maps["flow_fraction"][key] = float(fractions[state_index])
            block_maps["flow_count"][key] = float(observation.counts[state_index])
    for observation in counts:
        if observation.gate_index is None:
            key = f"{observation.condition}|week{observation.week}|rep={observation.replicate_id}"
            block_maps["count_total"][key] = float(observation.value)
        else:
            key = (
                f"{observation.condition}|week{observation.week}|gate={cfg.STATE_NAMES[observation.gate_index]}"
                f"|rep={observation.replicate_id}"
            )
            block_maps["count_gate"][key] = float(observation.value)
    for observation in qpcdr:
        key = (
            f"{observation.condition}|week{observation.week}|state={cfg.STATE_NAMES[observation.gate_index]}"
            f"|species={cfg.SPECIES[observation.species_index]}|batch={structure.qpcdr_batches[observation.batch_index]}|rep={observation.replicate_id}"
        )
        block_maps["qpcdr"][key] = float(observation.value)
    for observation in ectag_hist:
        prefix = f"{observation.condition}|week{observation.week}|state={cfg.STATE_NAMES[observation.gate_index]}|species={cfg.SPECIES[observation.species_index]}|rep={observation.replicate_id}"
        probs = observation.counts / max(float(np.sum(observation.counts)), 1e-12)
        for bin_index, probability in enumerate(probs.tolist()):
            block_maps["ectag_hist"][f"{prefix}|bin={bin_index}"] = float(probability)
        block_maps["ectag_moments"][f"{prefix}|zero_fraction"] = float(probs[0])
        block_maps["ectag_moments"][f"{prefix}|tail_ge_8"] = structure.binning.tail_probability(probs, 8)
        block_maps["ectag_moments"][f"{prefix}|tail_ge_16"] = structure.binning.tail_probability(probs, 16)
    for observation in ectag_corr:
        key = (
            f"{observation.condition}|week{observation.week}|state={cfg.STATE_NAMES[observation.gate_index]}"
            f"|pair={cfg.SPECIES[observation.species_a]}-{cfg.SPECIES[observation.species_b]}|rep={observation.replicate_id}"
        )
        block_maps["ectag_corr"][key] = float(observation.correlation)
    return SummaryCollection.from_block_maps(block_maps)


def _empty_v4_lite_block_maps() -> dict[str, dict[str, float]]:
    return {
        "flow_fraction": {},
        "flow_count": {},
        "count_total": {},
        "count_gate": {},
        "qpcdr": {},
        "ectag_hist": {},
        "ectag_moments": {},
        "ectag_corr": {},
    }


def summarize_dataset_v4_lite(
    dataset: CanonicalFitDataset,
    *,
    condition_names: Iterable[str] | None = None,
    binning: CopyNumberBinning | None = None,
    dynamic_only: bool = True,
) -> SummaryCollection:
    structure = V4LiteStructure.default(qpcdr_batches=dataset.qpcdr_batches())
    if binning is not None:
        structure = V4LiteStructure(structure.transition_edges, binning=binning, qpcdr_batches=structure.qpcdr_batches)
    tensor = build_v4_lite_tensor(dataset, condition_names=condition_names, structure=structure)
    if dynamic_only:
        return tensor.observed_summary
    return tensor.observed_summary


def _copy_means(distributions: np.ndarray, binning: CopyNumberBinning) -> np.ndarray:
    means = np.zeros((cfg.N_STATES, cfg.N_SPECIES), dtype=float)
    for state_index in range(cfg.N_STATES):
        for species_index in range(cfg.N_SPECIES):
            means[state_index, species_index] = binning.mean(distributions[state_index, species_index, :])
    return means


def _copy_log_signals(distributions: np.ndarray, binning: CopyNumberBinning) -> np.ndarray:
    values = np.asarray(distributions, dtype=float).reshape(cfg.N_STATES, cfg.N_SPECIES, binning.n_bins)
    log_centers = np.log1p(np.asarray(binning.centers, dtype=float))
    return np.tensordot(values, log_centers, axes=([2], [0]))


def _drug_adjusted_log_signals(log_signals: np.ndarray, exposure_C: float, exposure_P: float) -> np.ndarray:
    adjusted = np.asarray(log_signals, dtype=float).copy()
    adjusted[:, cfg.CDK4] = adjusted[:, cfg.CDK4] / (1.0 + max(float(exposure_C), 0.0))
    adjusted[:, cfg.PDGFRA] = adjusted[:, cfg.PDGFRA] / (1.0 + max(float(exposure_P), 0.0))
    return adjusted


def _gate_mixture_weights(abundance: np.ndarray, purity: np.ndarray, gate_index: int) -> np.ndarray:
    weights = purity[gate_index, :] * np.asarray(abundance, dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.eye(cfg.N_STATES, dtype=float)[gate_index]
    return weights / total


def _copy_number_kernel(
    params: V4LiteParameters,
    structure: V4LiteStructure,
    state_index: int,
    species_index: int,
    exposure_C: float,
    exposure_P: float,
    burden: float,
    model_version: str,
) -> np.ndarray:
    up_logit = params.kernel_up_species[species_index]
    down_logit = params.kernel_down_species[species_index]
    if model_version in {"M1", "M2", "M3"}:
        up_logit += params.kernel_up_state[state_index]
        down_logit += params.kernel_down_state[state_index]
    if species_index == cfg.CDK4:
        down_logit += params.kernel_down_C_target * exposure_C
    if species_index == cfg.PDGFRA:
        down_logit += params.kernel_down_P_target * exposure_P
    if model_version == "M2":
        down_logit += params.burden_loss_effect * burden
    stay_probability, down_probability, up_probability = _softmax(np.array([0.0, down_logit, up_logit], dtype=float))
    kernel = np.zeros((structure.binning.n_bins, structure.binning.n_bins), dtype=float)
    for bin_index in range(structure.binning.n_bins):
        kernel[bin_index, bin_index] += stay_probability
        kernel[bin_index, max(0, bin_index - 1)] += down_probability
        kernel[bin_index, min(structure.binning.n_bins - 1, bin_index + 1)] += up_probability
    return kernel


def _state_transition_matrix(
    params: V4LiteParameters,
    structure: V4LiteStructure,
    copy_distributions: np.ndarray,
    exposure_C: float,
    exposure_P: float,
    model_version: str,
) -> np.ndarray:
    q_generator = np.zeros((cfg.N_STATES, cfg.N_STATES), dtype=float)
    z = _drug_adjusted_log_signals(_copy_log_signals(copy_distributions, structure.binning), exposure_C, exposure_P)
    alpha = params.alpha_state - float(np.mean(params.alpha_state))
    potentials = np.zeros((cfg.N_STATES, cfg.N_STATES), dtype=float)
    for source in range(cfg.N_STATES):
        potentials[source, cfg.NPC] = alpha[cfg.NPC] + params.beta_C * z[source, cfg.CDK4]
        potentials[source, cfg.OPC] = alpha[cfg.OPC] + params.beta_P * z[source, cfg.PDGFRA]
        potentials[source, cfg.AC] = alpha[cfg.AC]
        potentials[source, cfg.MES] = alpha[cfg.MES]
    for source, target in structure.transition_edges:
        mobility = float(np.exp(np.clip(params.mobility_log[structure.mobility_index(source, target)], -12.0, 4.0)))
        score = np.log(max(mobility, 1e-12))
        if model_version in {"M1", "M2", "M3"}:
            score += potentials[source, target] - potentials[source, source]
            score += params.lambda_M * z[source, cfg.MYC]
        q_generator[source, target] = float(np.exp(np.clip(score, -30.0, 10.0)))
    for state_index in range(cfg.N_STATES):
        q_generator[state_index, state_index] = -float(np.sum(q_generator[state_index, :]))
    transition = expm(q_generator)
    transition = np.clip(np.asarray(transition, dtype=float), 1e-12, None)
    transition /= np.sum(transition, axis=1, keepdims=True)
    return transition


def _growth_rates(
    params: V4LiteParameters,
    structure: V4LiteStructure,
    copy_distributions: np.ndarray,
    abundance: np.ndarray,
    exposure_C: float,
    exposure_P: float,
    has_total_counts: bool,
    burden_star: float,
    model_version: str,
) -> np.ndarray:
    growth = np.asarray(params.growth_base, dtype=float).copy()
    if model_version in {"M1", "M2", "M3"}:
        z = _copy_log_signals(copy_distributions, structure.binning)
        adjusted = _drug_adjusted_log_signals(z, exposure_C, exposure_P)
        proliferative = 0.5 * (adjusted[:, cfg.MYC] + adjusted[:, cfg.CDK4])
        w_c = np.array([1.0, params.omega_O_given_C, 0.0, 0.0], dtype=float)
        w_p = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
        growth += params.theta_P * proliferative
        growth -= params.chi_C * exposure_C * z[:, cfg.CDK4] * w_c
        growth -= params.chi_P * exposure_P * z[:, cfg.PDGFRA] * w_p
    if model_version == "M2":
        burden = np.mean(_copy_log_signals(copy_distributions, structure.binning), axis=1)
        growth -= params.theta_B * np.square(burden - burden_star)
    if not has_total_counts:
        fractions = _normalize_simplex(abundance)
        growth -= float(np.dot(fractions, growth))
    return growth


def predict_v4_lite(
    tensor: V4LiteTensor,
    params: V4LiteParameters,
    *,
    model_version: str = "M1",
    dynamics_mode: str = "joint",
    frozen_copy_distributions: Mapping[str, np.ndarray] | None = None,
    empirical_abundance_proxy: Mapping[str, np.ndarray] | None = None,
    reference: SummaryCollection | None = None,
) -> V4LitePrediction:
    cfg.require(model_version in V4LiteModelVersion, f"Unknown v4-lite model version {model_version}.")
    cfg.require(dynamics_mode in V4LiteDynamicsMode, f"Unknown v4-lite dynamics mode {dynamics_mode}.")
    _validate_v4_lite_parameters(params, tensor.structure)
    n_weeks = len(tensor.weeks)
    n_bins = tensor.structure.binning.n_bins
    state_abundance: dict[str, np.ndarray] = {}
    copy_distributions: dict[str, np.ndarray] = {}
    transition_matrices: dict[str, np.ndarray] = {}
    growth_rates: dict[str, np.ndarray] = {}
    copy_kernels: dict[str, np.ndarray] = {}
    for condition_name in tensor.condition_names:
        abundance = np.zeros((n_weeks, cfg.N_STATES), dtype=float)
        distributions = np.zeros((n_weeks, cfg.N_STATES, cfg.N_SPECIES, n_bins), dtype=float)
        transitions = np.zeros((max(0, n_weeks - 1), cfg.N_STATES, cfg.N_STATES), dtype=float)
        growth = np.zeros((max(0, n_weeks - 1), cfg.N_STATES), dtype=float)
        kernels = np.zeros((max(0, n_weeks - 1), cfg.N_STATES, cfg.N_SPECIES, n_bins, n_bins), dtype=float)
        abundance[0, :] = np.asarray(tensor.initial_state_abundance[condition_name], dtype=float)
        distributions[0, :, :, :] = np.asarray(tensor.initial_copy_distributions[condition_name], dtype=float)
        if dynamics_mode == "ecDNA_only":
            abundance[:, :] = abundance[0, :]
        if dynamics_mode == "ecDNA_only" and empirical_abundance_proxy is not None and condition_name in empirical_abundance_proxy:
            proxy = np.asarray(empirical_abundance_proxy[condition_name], dtype=float)
            cfg.require(proxy.shape == (n_weeks, cfg.N_STATES), f"empirical abundance proxy for {condition_name} has invalid shape.")
            abundance[:, :] = np.clip(proxy, 1e-12, None)
        if dynamics_mode == "state_only":
            cfg.require(frozen_copy_distributions is not None, "state_only prediction requires frozen_copy_distributions.")
            cfg.require(condition_name in frozen_copy_distributions, f"Missing frozen copy distributions for {condition_name}.")
            frozen = np.asarray(frozen_copy_distributions[condition_name], dtype=float)
            cfg.require(
                frozen.shape == (n_weeks, cfg.N_STATES, cfg.N_SPECIES, n_bins),
                f"frozen copy distributions for {condition_name} have invalid shape.",
            )
            distributions[:, :, :, :] = frozen
        for interval_index in range(n_weeks - 1):
            exposure_C = params.exposure_C_scale * float(tensor.exposure_C[condition_name][interval_index])
            exposure_P = params.exposure_P_scale * float(tensor.exposure_P[condition_name][interval_index])
            current_distributions = distributions[interval_index, :, :, :]
            burden = np.mean(_copy_log_signals(current_distributions, tensor.structure.binning), axis=1)
            after_kernel = np.zeros((cfg.N_STATES, cfg.N_SPECIES, n_bins), dtype=float)
            for source in range(cfg.N_STATES):
                for species_index in range(cfg.N_SPECIES):
                    kernel = _copy_number_kernel(
                        params,
                        tensor.structure,
                        source,
                        species_index,
                        exposure_C,
                        exposure_P,
                        float(burden[source]),
                        model_version,
                    )
                    kernels[interval_index, source, species_index, :, :] = kernel
                    after_kernel[source, species_index, :] = distributions[interval_index, source, species_index, :] @ kernel
            if dynamics_mode == "ecDNA_only":
                distributions[interval_index + 1, :, :, :] = after_kernel
                transitions[interval_index, :, :] = np.eye(cfg.N_STATES, dtype=float)
                growth[interval_index, :] = 0.0
                continue

            transition = _state_transition_matrix(params, tensor.structure, current_distributions, exposure_C, exposure_P, model_version)
            growth_vector = _growth_rates(
                params,
                tensor.structure,
                current_distributions,
                abundance[interval_index, :],
                exposure_C,
                exposure_P,
                tensor.has_total_counts,
                tensor.burden_star,
                model_version,
            )
            source_mass = abundance[interval_index, :] * np.exp(np.clip(growth_vector, -30.0, 30.0))
            contribution = source_mass[:, None] * transition
            abundance[interval_index + 1, :] = np.clip(np.sum(contribution, axis=0), 1e-12, None)
            if dynamics_mode == "joint":
                for target in range(cfg.N_STATES):
                    denominator = float(abundance[interval_index + 1, target])
                    for species_index in range(cfg.N_SPECIES):
                        mixture = np.zeros(n_bins, dtype=float)
                        for source in range(cfg.N_STATES):
                            mixture += contribution[source, target] * after_kernel[source, species_index, :]
                        distributions[interval_index + 1, target, species_index, :] = _normalize_simplex(mixture / max(denominator, 1e-12))
            transitions[interval_index, :, :] = transition
            growth[interval_index, :] = growth_vector
        state_abundance[condition_name] = abundance
        copy_distributions[condition_name] = distributions
        transition_matrices[condition_name] = transitions
        growth_rates[condition_name] = growth
        copy_kernels[condition_name] = kernels
    summary = _prediction_summary(tensor, params, state_abundance, copy_distributions, model_version)
    if reference is not None:
        summary = summary.align_to(reference)
    return V4LitePrediction(
        condition_names=tensor.condition_names,
        weeks=tensor.weeks,
        state_abundance=state_abundance,
        copy_distributions=copy_distributions,
        transition_matrices=transition_matrices,
        growth_rates=growth_rates,
        copy_kernels=copy_kernels,
        summary=summary,
    )


def _expected_gate_distribution(
    tensor: V4LiteTensor,
    params: V4LiteParameters,
    prediction: V4LitePrediction,
    condition_name: str,
    week: int,
    gate_index: int,
    species_index: int,
) -> np.ndarray:
    week_index = tensor.week_to_index[week]
    abundance = prediction.state_abundance[condition_name][week_index, :]
    weights = _gate_mixture_weights(abundance, params.sort_purity_matrix, gate_index)
    distribution = np.zeros(tensor.structure.binning.n_bins, dtype=float)
    for state_index in range(cfg.N_STATES):
        distribution += weights[state_index] * prediction.copy_distributions[condition_name][week_index, state_index, species_index, :]
    return _normalize_simplex(distribution)


def _expected_gate_mean(
    tensor: V4LiteTensor,
    params: V4LiteParameters,
    prediction: V4LitePrediction,
    condition_name: str,
    week: int,
    gate_index: int,
    species_index: int,
) -> float:
    distribution = _expected_gate_distribution(tensor, params, prediction, condition_name, week, gate_index, species_index)
    return tensor.structure.binning.mean(distribution)


def _expected_qpcdr_value(
    tensor: V4LiteTensor,
    params: V4LiteParameters,
    prediction: V4LitePrediction,
    observation: QPCDRObservation,
) -> float:
    mean_copy = _expected_gate_mean(tensor, params, prediction, observation.condition, observation.week, observation.gate_index, observation.species_index)
    batch_offset = params.qpcdr_batch_offsets[observation.batch_index]
    if tensor.dataset.qpcdr_scale() == "ct":
        return float(params.qpcdr_intercept[observation.species_index] + batch_offset - params.qpcdr_slope[observation.species_index] * np.log10(mean_copy + 1e-6))
    return float(params.qpcdr_intercept[observation.species_index] + batch_offset + params.qpcdr_slope[observation.species_index] * mean_copy)


def _prediction_summary(
    tensor: V4LiteTensor,
    params: V4LiteParameters,
    state_abundance: Mapping[str, np.ndarray],
    copy_distributions: Mapping[str, np.ndarray],
    model_version: str,
) -> SummaryCollection:
    block_maps = _empty_v4_lite_block_maps()
    week_to_index = tensor.week_to_index
    for observation in tensor.flow_observations:
        week_index = week_to_index[observation.week]
        abundance = state_abundance[observation.condition][week_index, :]
        fractions = _normalize_simplex(abundance)
        observed_fractions = _normalize_simplex(params.sort_purity_matrix @ fractions)
        for state_index, state_name in enumerate(cfg.STATE_NAMES):
            key = f"{observation.condition}|week{observation.week}|state={state_name}|rep={observation.replicate_id}"
            block_maps["flow_fraction"][key] = float(observed_fractions[state_index])
            gate_count = float(np.sum(params.sort_purity_matrix[state_index, :] * abundance))
            block_maps["flow_count"][key] = gate_count
    for observation in tensor.count_observations:
        week_index = week_to_index[observation.week]
        abundance = state_abundance[observation.condition][week_index, :]
        if observation.gate_index is None:
            key = f"{observation.condition}|week{observation.week}|rep={observation.replicate_id}"
            block_maps["count_total"][key] = float(np.sum(abundance))
        else:
            key = (
                f"{observation.condition}|week{observation.week}|gate={cfg.STATE_NAMES[observation.gate_index]}"
                f"|rep={observation.replicate_id}"
            )
            block_maps["count_gate"][key] = float(np.sum(params.sort_purity_matrix[observation.gate_index, :] * abundance))
    fake_prediction = V4LitePrediction(
        condition_names=tensor.condition_names,
        weeks=tensor.weeks,
        state_abundance=dict(state_abundance),
        copy_distributions=dict(copy_distributions),
        transition_matrices={},
        growth_rates={},
        copy_kernels={},
        summary=SummaryCollection({}),
    )
    for observation in tensor.qpcdr_observations:
        key = (
            f"{observation.condition}|week{observation.week}|state={cfg.STATE_NAMES[observation.gate_index]}"
            f"|species={cfg.SPECIES[observation.species_index]}|batch={tensor.structure.qpcdr_batches[observation.batch_index]}|rep={observation.replicate_id}"
        )
        block_maps["qpcdr"][key] = _expected_qpcdr_value(tensor, params, fake_prediction, observation)
    for observation in tensor.ectag_hist_observations:
        probs = _expected_gate_distribution(
            tensor,
            params,
            fake_prediction,
            observation.condition,
            observation.week,
            observation.gate_index,
            observation.species_index,
        )
        prefix = f"{observation.condition}|week{observation.week}|state={cfg.STATE_NAMES[observation.gate_index]}|species={cfg.SPECIES[observation.species_index]}|rep={observation.replicate_id}"
        for bin_index, probability in enumerate(probs.tolist()):
            block_maps["ectag_hist"][f"{prefix}|bin={bin_index}"] = float(probability)
        block_maps["ectag_moments"][f"{prefix}|zero_fraction"] = float(probs[0])
        block_maps["ectag_moments"][f"{prefix}|tail_ge_8"] = tensor.structure.binning.tail_probability(probs, 8)
        block_maps["ectag_moments"][f"{prefix}|tail_ge_16"] = tensor.structure.binning.tail_probability(probs, 16)
    for observation in tensor.ectag_corr_observations:
        expected = params.co_segregation_rho if model_version == "M3" else 0.0
        key = (
            f"{observation.condition}|week{observation.week}|state={cfg.STATE_NAMES[observation.gate_index]}"
            f"|pair={cfg.SPECIES[observation.species_a]}-{cfg.SPECIES[observation.species_b]}|rep={observation.replicate_id}"
        )
        block_maps["ectag_corr"][key] = float(expected)
    return SummaryCollection.from_block_maps(block_maps)


def _validate_v4_lite_parameters(params: V4LiteParameters, structure: V4LiteStructure) -> None:
    array_shapes = {
        "qpcdr_intercept": (cfg.N_SPECIES,),
        "qpcdr_slope": (cfg.N_SPECIES,),
        "qpcdr_sigma": (cfg.N_SPECIES,),
        "qpcdr_batch_offsets": (structure.n_qpcdr_batches,),
        "ectag_concentration": (cfg.N_SPECIES,),
        "kernel_up_species": (cfg.N_SPECIES,),
        "kernel_down_species": (cfg.N_SPECIES,),
        "kernel_up_state": (cfg.N_STATES,),
        "kernel_down_state": (cfg.N_STATES,),
        "alpha_state": (cfg.N_STATES,),
        "mobility_log": (structure.n_mobility_edges,),
        "growth_base": (cfg.N_STATES,),
    }
    for field_name, shape in array_shapes.items():
        values = np.asarray(getattr(params, field_name), dtype=float)
        cfg.require(values.shape == shape, f"{field_name} must have shape {shape}.")
        cfg.require(np.all(np.isfinite(values)), f"{field_name} must be finite.")
    for field_name in ("qpcdr_slope", "qpcdr_sigma", "ectag_concentration"):
        cfg.require(np.all(np.asarray(getattr(params, field_name), dtype=float) > 0.0), f"{field_name} must be positive.")
    for field_name in (
        "qpcdr_df",
        "flow_concentration",
        "count_dispersion",
        "count_gate_dispersion",
        "ectag_corr_sigma",
        "exposure_C_scale",
        "exposure_P_scale",
        "beta_C",
        "beta_P",
        "lambda_M",
        "chi_C",
        "chi_P",
    ):
        value = float(getattr(params, field_name))
        cfg.require(np.isfinite(value) and value > 0.0, f"{field_name} must be positive and finite.")
    cfg.require(0.0 <= params.omega_O_given_C <= 1.0, "omega_O_given_C must lie in [0, 1].")
    cfg.require(params.theta_B >= 0.0, "theta_B must be non-negative.")
    cfg.require(-0.95 <= params.co_segregation_rho <= 0.95, "co_segregation_rho must lie in [-0.95, 0.95].")
    _normalize_purity_matrix(params.sort_purity_matrix)


@dataclass(frozen=True)
class V4LiteFieldSpec:
    name: str
    group: str
    transform: str
    shape: tuple[int, ...]
    prior_center: np.ndarray
    prior_scale: np.ndarray
    versions: frozenset[str]
    shrinkage: bool = False
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    boundary_type: str = "soft"

    @property
    def raw_size(self) -> int:
        return int(np.prod(self.shape, dtype=int)) if self.shape else 1

    @property
    def unconstrained_size(self) -> int:
        if self.transform == "zero_sum":
            return cfg.LATENT_DIM
        if self.transform == "column_simplex":
            return cfg.N_STATES * (cfg.N_STATES - 1)
        return self.raw_size


def _infer_v4_lite_bounds(transform: str, center: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    center_values = np.asarray(center, dtype=float).reshape(-1)
    scale_values = np.maximum(np.asarray(scale, dtype=float).reshape(-1), 1e-8)
    if transform == "log":
        lower = np.full(center_values.size, 1e-12, dtype=float)
        upper = np.full(center_values.size, np.inf, dtype=float)
        return lower, upper, "hard"
    if transform == "logit":
        lower = np.full(center_values.size, 1e-8, dtype=float)
        upper = np.full(center_values.size, 1.0 - 1e-8, dtype=float)
        return lower, upper, "hard"
    if transform == "bounded_identity":
        lower = np.full(center_values.size, -0.95, dtype=float)
        upper = np.full(center_values.size, 0.95, dtype=float)
        return lower, upper, "hard"
    if transform == "column_simplex":
        lower = np.full(center_values.size, 1e-8, dtype=float)
        upper = np.full(center_values.size, 1.0, dtype=float)
        return lower, upper, "hard"
    lower = center_values - 4.0 * scale_values
    upper = center_values + 4.0 * scale_values
    return lower, upper, "soft"


def _field_specs(structure: V4LiteStructure) -> tuple[V4LiteFieldSpec, ...]:
    def spec(
        name: str,
        group: str,
        transform: str,
        shape: tuple[int, ...],
        center: float | np.ndarray,
        scale: float,
        versions: Iterable[str],
        *,
        shrinkage: bool = False,
        lower: float | np.ndarray | None = None,
        upper: float | np.ndarray | None = None,
        boundary_type: str | None = None,
    ) -> V4LiteFieldSpec:
        center_array = np.asarray(center, dtype=float)
        if not shape:
            center_array = center_array.reshape(1)
        else:
            center_array = np.broadcast_to(center_array, shape).astype(float).reshape(-1)
        scale_array = np.full(center_array.size, float(scale), dtype=float)
        if lower is None or upper is None:
            inferred_lower, inferred_upper, inferred_boundary_type = _infer_v4_lite_bounds(transform, center_array, scale_array)
        else:
            inferred_lower = np.broadcast_to(np.asarray(lower, dtype=float), center_array.shape).astype(float).reshape(-1)
            inferred_upper = np.broadcast_to(np.asarray(upper, dtype=float), center_array.shape).astype(float).reshape(-1)
            inferred_boundary_type = "hard"
        return V4LiteFieldSpec(
            name=name,
            group=group,
            transform=transform,
            shape=shape,
            prior_center=center_array,
            prior_scale=scale_array,
            versions=frozenset(versions),
            shrinkage=shrinkage,
            lower=inferred_lower,
            upper=inferred_upper,
            boundary_type=inferred_boundary_type if boundary_type is None else boundary_type,
        )

    all_versions = ("M0", "M1", "M2", "M3")
    coupled = ("M1", "M2", "M3")
    return (
        spec("qpcdr_intercept", "observation", "identity", (cfg.N_SPECIES,), 0.0, 0.75, all_versions),
        spec("qpcdr_slope", "observation", "log", (cfg.N_SPECIES,), 1.0, 0.35, all_versions),
        spec("qpcdr_sigma", "observation", "log", (cfg.N_SPECIES,), 0.25, 0.35, all_versions),
        spec("qpcdr_batch_offsets", "observation", "identity", (structure.n_qpcdr_batches,), 0.0, 0.25, all_versions, shrinkage=True),
        spec("flow_concentration", "observation", "log", (), 250.0, 0.40, all_versions),
        spec("count_dispersion", "observation", "log", (), 25.0, 0.50, all_versions),
        spec("count_gate_dispersion", "observation", "log", (), 25.0, 0.50, all_versions),
        spec("ectag_concentration", "observation", "log", (cfg.N_SPECIES,), 120.0, 0.45, all_versions),
        spec("ectag_corr_sigma", "observation", "log", (), 0.20, 0.35, all_versions),
        spec("exposure_C_scale", "exposure", "log", (), 1.0, 0.45, all_versions),
        spec("exposure_P_scale", "exposure", "log", (), 1.0, 0.45, all_versions),
        spec("kernel_up_species", "ecDNA_kernel", "identity", (cfg.N_SPECIES,), -2.20, 0.60, all_versions),
        spec("kernel_down_species", "ecDNA_kernel", "identity", (cfg.N_SPECIES,), -2.30, 0.60, all_versions),
        spec("kernel_up_state", "ecDNA_kernel", "identity", (cfg.N_STATES,), 0.0, 0.35, ("M2", "M3"), shrinkage=True),
        spec("kernel_down_state", "ecDNA_kernel", "identity", (cfg.N_STATES,), 0.0, 0.35, ("M2", "M3"), shrinkage=True),
        spec("kernel_down_C_target", "ecDNA_kernel", "identity", (), 0.0, 0.35, all_versions, shrinkage=True),
        spec("kernel_down_P_target", "ecDNA_kernel", "identity", (), 0.0, 0.35, all_versions, shrinkage=True),
        spec("alpha_state", "state_abundance", "zero_sum", (cfg.N_STATES,), 0.0, 0.50, coupled),
        spec("beta_C", "state_abundance", "log", (), 0.50, 0.40, coupled),
        spec("beta_P", "state_abundance", "log", (), 0.50, 0.40, coupled),
        spec("lambda_M", "state_abundance", "log", (), 0.20, 0.45, coupled),
        spec("mobility_log", "state_abundance", "identity", (structure.n_mobility_edges,), -2.0, 0.70, all_versions),
        spec("growth_base", "state_abundance", "identity", (cfg.N_STATES,), 0.0, 0.40, all_versions),
        spec("theta_P", "state_abundance", "identity", (), 0.0, 0.35, coupled, shrinkage=True),
        spec("chi_C", "state_abundance", "log", (), 0.10, 0.45, coupled),
        spec("chi_P", "state_abundance", "log", (), 0.10, 0.45, coupled),
        spec("omega_O_given_C", "state_abundance", "logit", (), 0.45, 0.35, coupled),
        spec("theta_B", "burden", "log", (), 0.05, 0.50, ("M2",), shrinkage=True),
        spec("burden_loss_effect", "burden", "identity", (), 0.0, 0.35, ("M2",), shrinkage=True),
        spec("co_segregation_rho", "co_segregation", "bounded_identity", (), 0.0, 0.25, ("M3",), shrinkage=True),
    )


def _simplex_to_unconstrained(values: np.ndarray) -> np.ndarray:
    simplex = _normalize_simplex(values)
    return np.log(np.clip(simplex[:-1], 1e-12, None) / np.clip(simplex[-1], 1e-12, None))


def _simplex_from_unconstrained(values: np.ndarray) -> np.ndarray:
    logits = np.concatenate([np.asarray(values, dtype=float).reshape(-1), np.array([0.0], dtype=float)])
    return _softmax(logits)


def _column_simplex_to_unconstrained(values: np.ndarray) -> np.ndarray:
    matrix = _normalize_purity_matrix(np.asarray(values, dtype=float).reshape(cfg.N_STATES, cfg.N_STATES))
    return np.concatenate([_simplex_to_unconstrained(matrix[:, column]) for column in range(cfg.N_STATES)], axis=0)


def _column_simplex_from_unconstrained(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(-1)
    pieces: list[np.ndarray] = []
    width = cfg.N_STATES - 1
    for column in range(cfg.N_STATES):
        pieces.append(_simplex_from_unconstrained(flat[column * width : (column + 1) * width]))
    return np.stack(pieces, axis=1)


def _zero_sum_to_unconstrained(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=float).reshape(cfg.N_STATES)
    centered = centered - float(np.mean(centered))
    return cfg.HELMERT_SUBMATRIX.T @ centered


def _zero_sum_from_unconstrained(values: np.ndarray) -> np.ndarray:
    return cfg.HELMERT_SUBMATRIX @ np.asarray(values, dtype=float).reshape(cfg.LATENT_DIM)


class V4LiteParameterAdapter:
    def __init__(
        self,
        *,
        structure: V4LiteStructure,
        base_params: V4LiteParameters,
        active_groups: Iterable[str],
        model_version: str,
    ):
        self.structure = structure
        self.base_params = base_params.copy()
        self.model_version = model_version
        groups = set(active_groups)
        self.specs = tuple(
            spec
            for spec in _field_specs(structure)
            if spec.group in groups and model_version in spec.versions
        )
        offset = 0
        slices: list[tuple[int, int]] = []
        for spec in self.specs:
            slices.append((offset, offset + spec.unconstrained_size))
            offset += spec.unconstrained_size
        self.slices = tuple(slices)
        self.dimension = offset

    def default_vector(self) -> np.ndarray:
        return self.pack(self.base_params)

    def pack(self, params: V4LiteParameters) -> np.ndarray:
        pieces = [self._to_unconstrained(spec, self._raw(params, spec)) for spec in self.specs]
        return np.concatenate(pieces, axis=0) if pieces else np.zeros(0, dtype=float)

    def unpack(self, vector: np.ndarray) -> V4LiteParameters:
        flat = np.asarray(vector, dtype=float).reshape(-1)
        cfg.require(flat.size == self.dimension, f"Expected vector dimension {self.dimension}, got {flat.size}.")
        params = self.base_params.copy()
        for spec, (start, stop) in zip(self.specs, self.slices):
            raw = self._from_unconstrained(spec, flat[start:stop])
            if not spec.shape:
                setattr(params, spec.name, float(raw[0]))
            else:
                setattr(params, spec.name, raw.reshape(spec.shape).copy())
        _validate_v4_lite_parameters(params, self.structure)
        return params

    def proposal_scales(self) -> np.ndarray:
        pieces: list[np.ndarray] = []
        for spec in self.specs:
            size = spec.unconstrained_size
            raw_scale = np.full(size, float(np.mean(spec.prior_scale)), dtype=float)
            pieces.append(raw_scale * (0.60 if spec.shrinkage else 1.0))
        return np.concatenate(pieces, axis=0) if pieces else np.zeros(0, dtype=float)

    def parameter_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for spec in self.specs:
            for index in range(spec.unconstrained_size):
                names.append(spec.name if spec.unconstrained_size == 1 else f"{spec.name}[{index}]")
        return tuple(names)

    def prior_penalty(self, params: V4LiteParameters) -> float:
        penalty = 0.0
        for spec in self.specs:
            raw = self._raw(params, spec)
            value = self._to_unconstrained(spec, raw)
            center = self._to_unconstrained(spec, self._center_for_spec(spec))
            scale = np.full(value.size, float(np.mean(spec.prior_scale)), dtype=float)
            z = (value - center) / np.maximum(scale, 1e-8)
            penalty += 0.5 * (2.0 if spec.shrinkage else 1.0) * float(np.dot(z, z)) / max(1, z.size)
        return float(penalty)

    def _center_for_spec(self, spec: V4LiteFieldSpec) -> np.ndarray:
        if spec.name == "mobility_log":
            return np.asarray(self._raw(self.base_params, spec), dtype=float)
        if spec.name == "sort_purity_matrix":
            return np.asarray(self.base_params.sort_purity_matrix, dtype=float).reshape(-1)
        if spec.name == "qpcdr_batch_offsets":
            return np.zeros(self.structure.n_qpcdr_batches, dtype=float)
        return spec.prior_center.copy()

    @staticmethod
    def _raw(params: V4LiteParameters, spec: V4LiteFieldSpec) -> np.ndarray:
        return np.asarray(getattr(params, spec.name), dtype=float).reshape(-1)

    @staticmethod
    def _to_unconstrained(spec: V4LiteFieldSpec, raw: np.ndarray) -> np.ndarray:
        values = np.asarray(raw, dtype=float).reshape(-1)
        if spec.transform in {"identity", "bounded_identity"}:
            return values
        if spec.transform == "log":
            return np.log(np.clip(values, 1e-12, None))
        if spec.transform == "logit":
            clipped = np.clip(values, 1e-8, 1.0 - 1e-8)
            return np.log(clipped / (1.0 - clipped))
        if spec.transform == "zero_sum":
            return _zero_sum_to_unconstrained(values)
        if spec.transform == "column_simplex":
            return _column_simplex_to_unconstrained(values)
        raise ValueError(f"Unsupported v4-lite transform {spec.transform}.")

    @staticmethod
    def _from_unconstrained(spec: V4LiteFieldSpec, values: np.ndarray) -> np.ndarray:
        flat = np.asarray(values, dtype=float).reshape(-1)
        if spec.transform in {"identity", "bounded_identity"}:
            return flat
        if spec.transform == "log":
            return np.exp(flat)
        if spec.transform == "logit":
            return np.asarray(cfg.sigmoid(flat), dtype=float).reshape(-1)
        if spec.transform == "zero_sum":
            return _zero_sum_from_unconstrained(flat)
        if spec.transform == "column_simplex":
            return _column_simplex_from_unconstrained(flat).reshape(-1)
        raise ValueError(f"Unsupported v4-lite transform {spec.transform}.")


class V4LiteObjective:
    def __init__(
        self,
        *,
        tensor: V4LiteTensor,
        active_groups: Iterable[str],
        model_version: str = "M1",
        base_params: V4LiteParameters | None = None,
        block_names: Iterable[str] | None = None,
        synthetic_observed_summary: SummaryCollection | None = None,
        dynamics_mode: str = "joint",
        heldout_weeks: Iterable[int] = (),
        frozen_copy_distributions: Mapping[str, np.ndarray] | None = None,
        empirical_abundance_proxy: Mapping[str, np.ndarray] | None = None,
    ):
        cfg.require(model_version in V4LiteModelVersion, f"Unknown v4-lite model version {model_version}.")
        cfg.require(dynamics_mode in V4LiteDynamicsMode, f"Unknown v4-lite dynamics mode {dynamics_mode}.")
        self.tensor = tensor
        self.active_groups = tuple(active_groups)
        self.model_version = model_version
        self.base_params = V4LiteParameters.default(tensor.structure, purity_matrix=tensor.dataset.purity_matrix, qpcdr_calibration=tensor.dataset.qpcdr_calibration) if base_params is None else base_params.copy()
        self.block_names = None if block_names is None else set(block_names)
        self.synthetic_observed_summary = synthetic_observed_summary
        self.observed_summary = tensor.observed_summary if synthetic_observed_summary is None else synthetic_observed_summary
        self.dynamics_mode = dynamics_mode
        self.heldout_weeks = frozenset(int(week) for week in heldout_weeks)
        self.frozen_copy_distributions = None if frozen_copy_distributions is None else {key: np.asarray(value, dtype=float).copy() for key, value in frozen_copy_distributions.items()}
        self.empirical_abundance_proxy = None if empirical_abundance_proxy is None else {key: np.asarray(value, dtype=float).copy() for key, value in empirical_abundance_proxy.items()}
        self.adapter = V4LiteParameterAdapter(
            structure=tensor.structure,
            base_params=self.base_params,
            active_groups=self.active_groups,
            model_version=model_version,
        )

    def with_synthetic_observed_summary(self, observed_summary: SummaryCollection) -> "V4LiteObjective":
        return V4LiteObjective(
            tensor=self.tensor,
            active_groups=self.active_groups,
            model_version=self.model_version,
            base_params=self.base_params,
            block_names=self.block_names,
            synthetic_observed_summary=observed_summary,
            dynamics_mode=self.dynamics_mode,
            heldout_weeks=self.heldout_weeks,
            frozen_copy_distributions=self.frozen_copy_distributions,
            empirical_abundance_proxy=self.empirical_abundance_proxy,
        )

    def evaluate_vector(self, vector: np.ndarray, *, return_artifacts: bool = False) -> V4LiteObjectiveResult:
        try:
            params = self.adapter.unpack(vector)
            prediction = predict_v4_lite(
                self.tensor,
                params,
                model_version=self.model_version,
                dynamics_mode=self.dynamics_mode,
                frozen_copy_distributions=self.frozen_copy_distributions,
                empirical_abundance_proxy=self.empirical_abundance_proxy,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return V4LiteObjectiveResult(INVALID_OBJECTIVE, INVALID_OBJECTIVE, 0.0, (), None)
        if self.synthetic_observed_summary is not None:
            block_results, data_nll = self._summary_likelihood(prediction)
        else:
            block_results, data_nll = self._raw_likelihood(params, prediction)
        prior_penalty = self.adapter.prior_penalty(params)
        total = float(data_nll + prior_penalty)
        artifacts = V4LiteObjectiveArtifacts(params=params.copy(), prediction=prediction, vector=np.asarray(vector, dtype=float).copy()) if return_artifacts else None
        return V4LiteObjectiveResult(total, float(data_nll), float(prior_penalty), tuple(block_results), artifacts)

    def _include_block(self, block_name: str) -> bool:
        return self.block_names is None or block_name in self.block_names

    def _include_week(self, week: int) -> bool:
        return int(week) not in self.heldout_weeks

    def _raw_likelihood(self, params: V4LiteParameters, prediction: V4LitePrediction) -> tuple[list[V4LiteBlockResult], float]:
        contributions: dict[str, list[tuple[float, float]]] = {name: [] for name in _empty_v4_lite_block_maps()}
        if self._include_block("flow_fraction") or self._include_block("flow_count"):
            for observation in self.tensor.flow_observations:
                if not self._include_week(observation.week):
                    continue
                probs = self._flow_probabilities(prediction, params, observation.condition, observation.week)
                logp = _dirichlet_multinomial_logpmf(observation.counts, probs, params.flow_concentration)
                residual = float(np.linalg.norm(observation.counts / max(observation.total, 1e-12) - probs))
                contributions["flow_fraction"].append((-logp, residual))
                contributions["flow_count"].append((-logp, float(abs(observation.total - observation.total))))
        if self._include_block("count_total"):
            for observation in self.tensor.count_observations:
                if observation.gate_index is not None:
                    continue
                if not self._include_week(observation.week):
                    continue
                mu = float(np.sum(prediction.state_abundance[observation.condition][self.tensor.week_to_index[observation.week], :]))
                logp = _negative_binomial_logpmf(observation.value, mu, params.count_dispersion)
                contributions["count_total"].append((-logp, float(abs(observation.value - mu))))
        if self._include_block("count_gate"):
            for observation in self.tensor.count_observations:
                if observation.gate_index is None:
                    continue
                if not self._include_week(observation.week):
                    continue
                abundance = prediction.state_abundance[observation.condition][self.tensor.week_to_index[observation.week], :]
                mu = float(np.sum(params.sort_purity_matrix[observation.gate_index, :] * abundance))
                logp = _negative_binomial_logpmf(observation.value, mu, params.count_gate_dispersion)
                contributions["count_gate"].append((-logp, float(abs(observation.value - mu))))
        if self._include_block("qpcdr"):
            for observation in self.tensor.qpcdr_observations:
                if not self._include_week(observation.week):
                    continue
                expected = _expected_qpcdr_value(self.tensor, params, prediction, observation)
                sigma = params.qpcdr_sigma[observation.species_index]
                logp = _student_t_logpdf(observation.value, expected, sigma, params.qpcdr_df)
                contributions["qpcdr"].append((-logp, float(abs(observation.value - expected))))
        if self._include_block("ectag_hist") or self._include_block("ectag_moments"):
            for observation in self.tensor.ectag_hist_observations:
                if not self._include_week(observation.week):
                    continue
                probs = _expected_gate_distribution(
                    self.tensor,
                    params,
                    prediction,
                    observation.condition,
                    observation.week,
                    observation.gate_index,
                    observation.species_index,
                )
                logp = _dirichlet_multinomial_logpmf(observation.counts, probs, params.ectag_concentration[observation.species_index])
                observed_probs = observation.counts / max(float(np.sum(observation.counts)), 1e-12)
                contributions["ectag_hist"].append((-logp, float(np.linalg.norm(observed_probs - probs))))
                moment_residual = abs(float(observed_probs[0]) - float(probs[0]))
                moment_residual += abs(self.tensor.structure.binning.tail_probability(observed_probs, 8) - self.tensor.structure.binning.tail_probability(probs, 8))
                contributions["ectag_moments"].append((-logp, moment_residual))
        if self._include_block("ectag_corr"):
            for observation in self.tensor.ectag_corr_observations:
                if not self._include_week(observation.week):
                    continue
                expected = params.co_segregation_rho if self.model_version == "M3" else 0.0
                sigma = params.ectag_corr_sigma / np.sqrt(max(1, observation.n_cells - 1))
                logp = _student_t_logpdf(observation.correlation, expected, sigma, 5.0)
                contributions["ectag_corr"].append((-logp, float(abs(observation.correlation - expected))))
        return self._block_results_from_contributions(contributions)

    def _summary_likelihood(self, prediction: V4LitePrediction) -> tuple[list[V4LiteBlockResult], float]:
        predicted = prediction.summary.align_to(self.observed_summary)
        contributions: dict[str, list[tuple[float, float]]] = {name: [] for name in _empty_v4_lite_block_maps()}
        for block_name in self.observed_summary.block_names():
            if not self._include_block(block_name):
                continue
            residual = predicted.blocks[block_name].values - self.observed_summary.blocks[block_name].values
            scale = max(float(np.std(self.observed_summary.blocks[block_name].values)), 1e-3)
            nll = 0.5 * (np.square(residual / scale) + np.log(2.0 * np.pi * scale * scale))
            for term, value in zip(nll.tolist(), np.abs(residual).tolist()):
                contributions[block_name].append((float(term), float(value)))
        return self._block_results_from_contributions(contributions)

    def _block_results_from_contributions(self, contributions: dict[str, list[tuple[float, float]]]) -> tuple[list[V4LiteBlockResult], float]:
        block_results: list[V4LiteBlockResult] = []
        total = 0.0
        for block_name, values in contributions.items():
            if not values:
                continue
            nll_values = np.array([value[0] for value in values], dtype=float)
            residual_values = np.array([value[1] for value in values], dtype=float)
            block_nll = float(np.mean(nll_values))
            total += block_nll
            block_results.append(
                V4LiteBlockResult(
                    name=block_name,
                    dimension=len(values),
                    negative_log_likelihood=block_nll,
                    residual_norm=float(np.sqrt(np.mean(np.square(residual_values)))),
                )
            )
        return block_results, float(total)

    def _flow_probabilities(self, prediction: V4LitePrediction, params: V4LiteParameters, condition_name: str, week: int) -> np.ndarray:
        abundance = prediction.state_abundance[condition_name][self.tensor.week_to_index[week], :]
        fractions = _normalize_simplex(abundance)
        return _normalize_simplex(params.sort_purity_matrix @ fractions)


def calibrate_v4_lite_observation_params(tensor: V4LiteTensor, base_params: V4LiteParameters) -> tuple[V4LiteParameters, dict[str, object]]:
    params = base_params.copy()
    report: dict[str, object] = {"qpcdr": {}, "ectag": {}, "flow": {}, "count": {}}

    q_groups: dict[tuple[int, int, str, int, int], list[float]] = {}
    batch_values: dict[int, list[float]] = {}
    for observation in tensor.qpcdr_observations:
        key = (observation.species_index, observation.batch_index, observation.condition, observation.week, observation.gate_index)
        q_groups.setdefault(key, []).append(float(observation.value))
        batch_values.setdefault(observation.batch_index, []).append(float(observation.value))
    q_sigma: dict[int, list[float]] = {index: [] for index in range(cfg.N_SPECIES)}
    for (species_index, _batch_index, _condition, _week, _gate), values in q_groups.items():
        if len(values) > 1:
            q_sigma[species_index].append(float(np.std(values, ddof=1)))
    preserved = set()
    for species_name, calibration in tensor.dataset.qpcdr_calibration.items():
        if species_name in cfg.SPECIES_INDEX and "sigma" in calibration:
            preserved.add(cfg.SPECIES_INDEX[species_name])
    insufficient_q: list[str] = []
    for species_index in range(cfg.N_SPECIES):
        if species_index in preserved:
            continue
        values = [value for value in q_sigma[species_index] if np.isfinite(value) and value > 0.0]
        if values:
            params.qpcdr_sigma[species_index] = max(float(np.median(values)), 1e-6)
        else:
            insufficient_q.append(cfg.SPECIES[species_index])
    if len(batch_values) > 1:
        batch_means = np.array([np.mean(batch_values.get(index, [0.0])) for index in range(tensor.structure.n_qpcdr_batches)], dtype=float)
        params.qpcdr_batch_offsets = batch_means - float(np.mean(batch_means))
    report["qpcdr"] = {
        "sigma": params.qpcdr_sigma.tolist(),
        "batch_offsets": params.qpcdr_batch_offsets.tolist(),
        "insufficient_replicates": insufficient_q,
        "provided_calibration_preserved": sorted(tensor.dataset.qpcdr_calibration),
    }

    hist_by_group: dict[tuple[str, int, int, int], list[np.ndarray]] = {}
    for observation in tensor.ectag_hist_observations:
        total = float(np.sum(observation.counts))
        if total <= 0.0:
            continue
        key = (observation.condition, observation.week, observation.gate_index, observation.species_index)
        hist_by_group.setdefault(key, []).append(observation.counts.astype(float) / total)
    ectag_alpha: dict[int, list[float]] = {index: [] for index in range(cfg.N_SPECIES)}
    insufficient_ectag: list[str] = []
    for (_condition, _week, _gate, species_index), rows in hist_by_group.items():
        if len(rows) < 2:
            insufficient_ectag.append(cfg.SPECIES[species_index])
            continue
        matrix = np.stack(rows, axis=0)
        mean_p = np.clip(np.mean(matrix, axis=0), 1e-6, 1.0)
        var_p = np.var(matrix, axis=0, ddof=1)
        usable = var_p > 1e-8
        if np.any(usable):
            alpha_estimates = mean_p[usable] * (1.0 - mean_p[usable]) / var_p[usable] - 1.0
            alpha_estimates = alpha_estimates[np.isfinite(alpha_estimates) & (alpha_estimates > 0.0)]
            if alpha_estimates.size:
                ectag_alpha[species_index].append(float(np.median(alpha_estimates)))
    for species_index in range(cfg.N_SPECIES):
        values = ectag_alpha[species_index]
        if values:
            params.ectag_concentration[species_index] = float(np.clip(np.median(values), 5.0, 5000.0))
    report["ectag"] = {
        "concentration": params.ectag_concentration.tolist(),
        "insufficient_replicates": sorted(set(insufficient_ectag)),
    }

    flow_by_week: dict[tuple[str, int], list[np.ndarray]] = {}
    for observation in tensor.flow_observations:
        if observation.total > 0.0:
            flow_by_week.setdefault((observation.condition, observation.week), []).append(observation.counts / observation.total)
    flow_alphas: list[float] = []
    for rows in flow_by_week.values():
        if len(rows) < 2:
            continue
        matrix = np.stack(rows, axis=0)
        pbar = np.clip(np.mean(matrix, axis=0), 1e-6, 1.0)
        var = np.var(matrix, axis=0, ddof=1)
        usable = var > 1e-8
        if np.any(usable):
            estimates = pbar[usable] * (1.0 - pbar[usable]) / var[usable] - 1.0
            estimates = estimates[np.isfinite(estimates) & (estimates > 0.0)]
            if estimates.size:
                flow_alphas.append(float(np.median(estimates)))
    if flow_alphas:
        params.flow_concentration = float(np.clip(np.median(flow_alphas), 5.0, 50000.0))
    report["flow"] = {
        "concentration": params.flow_concentration,
        "insufficient_replicates": not bool(flow_alphas),
    }

    count_by_week: dict[tuple[str, int, int | None], list[float]] = {}
    for observation in tensor.count_observations:
        count_by_week.setdefault((observation.condition, observation.week, observation.gate_index), []).append(float(observation.value))
    total_dispersions: list[float] = []
    gate_dispersions: list[float] = []
    count_replicate_flags = {"total": False, "gate": False}
    for (_condition, _week, gate_index), values in count_by_week.items():
        if len(values) < 2:
            continue
        gate_key = "gate" if gate_index is not None else "total"
        count_replicate_flags[gate_key] = True
        mean_value = float(np.mean(values))
        variance = float(np.var(values, ddof=1))
        if variance > mean_value > 0.0:
            estimate = float(mean_value * mean_value / (variance - mean_value))
            if gate_index is None:
                total_dispersions.append(estimate)
            else:
                gate_dispersions.append(estimate)
    if total_dispersions:
        params.count_dispersion = float(np.clip(np.median(total_dispersions), 1.0, 1e6))
    if gate_dispersions:
        params.count_gate_dispersion = float(np.clip(np.median(gate_dispersions), 1.0, 1e6))
    report["count"] = {
        "dispersion": params.count_dispersion,
        "gate_dispersion": params.count_gate_dispersion,
        "insufficient_replicates": not bool(total_dispersions or gate_dispersions),
        "total_insufficient_replicates": not count_replicate_flags["total"],
        "gate_insufficient_replicates": not count_replicate_flags["gate"],
    }
    return params, report


@dataclass(frozen=True)
class V4LiteStageDefinition:
    name: str
    active_groups: tuple[str, ...]
    block_names: tuple[str, ...] | None
    description: str
    model_version: str | None = None
    dynamics_mode: str = "joint"
    optional: bool = False


@dataclass(frozen=True)
class V4LiteOptimizationSettings:
    n_restarts: int = 4
    maxiter: int = 80
    random_start_scale: float = 1.0
    prior_predictive_draws: int = 16
    prior_predictive_seed: int = 17
    min_prior_predictive_pass_rate: float = 0.25
    profile_points: int = 5
    max_profile_dimensions: int = 8
    profile_maxiter: int = 25
    fake_recovery_restarts: int = 2
    min_objective_improvement: float = 1e-7
    max_posterior_predictive_relative_rmse: float = 3.0
    min_profile_objective_span: float = 1e-5
    require_fake_data_recovery_pass: bool = False
    posterior_draws: int = 32
    posterior_burnin: int = 16
    max_hmc_dimensions: int = 24
    sbc_datasets: int = 0
    loo_restarts: int = 1
    model_comparison_restarts: int = 2
    run_purity_sensitivity: bool = True
    write_optional_plots: bool = True
    seed: int = 123


@dataclass
class V4LiteStageFitResult:
    stage_name: str
    active_groups: tuple[str, ...]
    block_names: tuple[str, ...]
    objective_before: float | None
    objective_after: float | None
    active_parameter_names: tuple[str, ...]
    diagnostics: dict[str, object] = field(default_factory=dict)
    skipped_reason: str | None = None
    accepted: bool = False
    rejection_reasons: tuple[str, ...] = ()
    best_vector: np.ndarray | None = None
    best_params: V4LiteParameters | None = None


@dataclass
class V4LiteFitResult:
    prior_predictive: V4LitePriorPredictiveReport | None
    stage_results: list[V4LiteStageFitResult]
    final_params: V4LiteParameters
    tensor: V4LiteTensor
    posterior_samples: V4LitePosteriorSamples | None
    parameter_status_table: tuple[dict[str, object], ...]
    reports: V4LiteReports
    model_comparison: dict[str, float]
    ecDNA_reference_prediction: V4LitePrediction | None = None
    state_reference_prediction: V4LitePrediction | None = None
    model_fit_results: dict[str, dict[str, object]] = field(default_factory=dict)
    projection_targets: FullToLiteProjection | None = None


V4_LITE_STAGE_SEQUENCE = (
    V4LiteStageDefinition("observation", ("observation",), None, "Assay calibration and block noise."),
    V4LiteStageDefinition("week1-init-check", (), None, "Check week1 initialization.", optional=True),
    V4LiteStageDefinition("ecDNA-only", ("exposure", "ecDNA_kernel"), ("qpcdr", "ectag_hist", "ectag_moments", "ectag_corr"), "State-specific weekly ecDNA kernel.", dynamics_mode="ecDNA_only"),
    V4LiteStageDefinition("state-only", ("exposure", "state_abundance"), ("flow_fraction", "flow_count", "count_total", "count_gate"), "Weekly net growth and sparse state switching.", dynamics_mode="state_only"),
    V4LiteStageDefinition("joint-M1", ("exposure", "ecDNA_kernel", "state_abundance"), None, "Joint M1 v4-lite refinement.", model_version="M1"),
    V4LiteStageDefinition("M0-null", ("exposure", "ecDNA_kernel", "state_abundance"), None, "M0 null model comparison.", model_version="M0", optional=True),
    V4LiteStageDefinition("M2-burden", ("exposure", "ecDNA_kernel", "state_abundance", "burden"), None, "M2 burden/stress-lite extension.", model_version="M2", optional=True),
    V4LiteStageDefinition("M3-co-segregation", ("exposure", "ecDNA_kernel", "state_abundance", "co_segregation"), None, "M3 co-segregation extension.", model_version="M3", optional=True),
)


class V4LiteFitRunner:
    def __init__(
        self,
        dataset: CanonicalFitDataset,
        *,
        model_version: str = "M1",
        structure: V4LiteStructure | None = None,
        initial_params: V4LiteParameters | None = None,
        optimization_settings: V4LiteOptimizationSettings | None = None,
        output_dir: str | Path | None = None,
        condition_names: Iterable[str] | None = None,
        purity_sensitivity: Iterable[np.ndarray] | None = None,
    ):
        cfg.require(model_version in V4LiteModelVersion, f"Unknown v4-lite model version {model_version}.")
        self.model_version = model_version
        self.tensor = build_v4_lite_tensor(dataset, condition_names=condition_names, structure=structure)
        self.structure = self.tensor.structure
        base_params = V4LiteParameters.default(
            self.structure,
            purity_matrix=dataset.purity_matrix,
            qpcdr_calibration=dataset.qpcdr_calibration,
        )
        self.current_params = base_params if initial_params is None else initial_params.copy()
        self.settings = V4LiteOptimizationSettings() if optimization_settings is None else optimization_settings
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.purity_sensitivity = tuple(dataset.purity_sensitivity if purity_sensitivity is None else tuple(purity_sensitivity))
        self.calibration_report: dict[str, object] = {}
        self.ecDNA_reference_prediction: V4LitePrediction | None = None
        self.state_reference_prediction: V4LitePrediction | None = None
        self.model_fit_results: dict[str, dict[str, object]] = {}

    def run_all_stages(self) -> V4LiteFitResult:
        prior_predictive = self._run_prior_predictive()
        stage_results: list[V4LiteStageFitResult] = []
        for stage in V4_LITE_STAGE_SEQUENCE:
            result = self.run_stage(stage)
            if result.accepted and result.best_params is not None and stage.model_version in (None, self.model_version, "M1"):
                if stage.name not in {"M0-null", "M2-burden", "M3-co-segregation"}:
                    self.current_params = result.best_params.copy()
                    if stage.name == "ecDNA-only":
                        self.ecDNA_reference_prediction = predict_v4_lite(
                            self.tensor,
                            self.current_params,
                            model_version=self.model_version,
                            dynamics_mode="ecDNA_only",
                            empirical_abundance_proxy=self._empirical_abundance_proxy(),
                        )
                    if stage.name == "state-only":
                        frozen = self._frozen_copy_distributions()
                        self.state_reference_prediction = predict_v4_lite(
                            self.tensor,
                            self.current_params,
                            model_version=self.model_version,
                            dynamics_mode="state_only",
                            frozen_copy_distributions=frozen,
                        )
            stage_results.append(result)
        final_objective = V4LiteObjective(
            tensor=self.tensor,
            active_groups=("exposure", "ecDNA_kernel", "state_abundance", "burden", "co_segregation"),
            model_version=self.model_version,
            base_params=self.current_params,
        )
        final_vector = final_objective.adapter.pack(self.current_params)
        final_eval = final_objective.evaluate_vector(final_vector, return_artifacts=True)
        posterior_samples = run_v4_lite_hmc(final_objective, final_vector, self.settings)
        profile = run_v4_lite_profile_likelihood(
            final_objective,
            final_vector,
            profile_scales=np.maximum(final_objective.adapter.proposal_scales(), 1e-6),
            n_points=self.settings.profile_points,
            max_dimensions=self.settings.max_profile_dimensions,
            profile_maxiter=self.settings.profile_maxiter,
        )
        fake_recovery = run_v4_lite_fake_data_recovery(
            final_objective,
            final_vector,
            self._optimize_objective,
            n_restarts=self.settings.fake_recovery_restarts,
        )
        loo = run_leave_one_week_out(final_objective, final_vector, self._optimize_objective, n_restarts=self.settings.loo_restarts)
        sbc_report = run_v4_lite_sbc(self, self.settings.sbc_datasets, self.model_version) if self.settings.sbc_datasets > 0 else None
        model_comparison, self.model_fit_results = self._run_model_comparison()
        parameter_status_table = build_parameter_status_table(final_objective, final_vector, profile, fake_recovery, posterior_samples)
        prior_diagnostics_report = build_prior_diagnostics_report(final_objective, final_vector)
        final_prediction = final_eval.artifacts.prediction if final_eval.artifacts is not None else predict_v4_lite(self.tensor, self.current_params, model_version=self.model_version)
        purity_sensitivity = self._purity_sensitivity_report(final_prediction) if self.settings.run_purity_sensitivity else ()
        reports = build_v4_lite_reports(
            self.tensor,
            final_prediction,
            stage_results,
            parameter_status_table,
            fake_recovery,
            loo,
            sbc_report,
            model_comparison,
            self.calibration_report,
            purity_sensitivity,
            prior_diagnostics_report,
        )
        if self.output_dir is not None:
            write_v4_lite_reports(self.output_dir, reports, parameter_status_table, model_comparison, write_optional_plots=self.settings.write_optional_plots)
        projection_targets = _projection_from_prediction(final_prediction)
        return V4LiteFitResult(
            prior_predictive=prior_predictive,
            stage_results=stage_results,
            final_params=self.current_params.copy(),
            tensor=self.tensor,
            posterior_samples=posterior_samples,
            parameter_status_table=parameter_status_table,
            reports=reports,
            model_comparison=model_comparison,
            ecDNA_reference_prediction=self.ecDNA_reference_prediction,
            state_reference_prediction=self.state_reference_prediction,
            model_fit_results=self.model_fit_results,
            projection_targets=projection_targets,
        )

    def run_stage(self, stage: V4LiteStageDefinition) -> V4LiteStageFitResult:
        stage_version = stage.model_version or self.model_version
        if stage.name == "observation":
            calibrated_params, report = calibrate_v4_lite_observation_params(self.tensor, self.current_params)
            self.calibration_report = report
            return V4LiteStageFitResult(
                stage.name,
                stage.active_groups,
                self._available_blocks(stage.block_names),
                None,
                None,
                ("qpcdr_sigma", "qpcdr_batch_offsets", "ectag_concentration", "flow_concentration", "count_dispersion", "count_gate_dispersion"),
                diagnostics={"calibration": report},
                accepted=True,
                best_params=calibrated_params,
            )
        if stage.name == "week1-init-check":
            return V4LiteStageFitResult(stage.name, stage.active_groups, (), None, None, (), {"initial_total": self._initial_totals()}, accepted=True)
        if stage.name == "M3-co-segregation" and not self.tensor.has_same_cell_ectag:
            return V4LiteStageFitResult(stage.name, stage.active_groups, (), None, None, (), skipped_reason="No same-cell multicolor ecTAG readout.", accepted=False)
        block_names = self._available_blocks(stage.block_names)
        if not block_names:
            return V4LiteStageFitResult(stage.name, stage.active_groups, (), None, None, (), skipped_reason="No observed blocks are available for this stage.")
        objective = V4LiteObjective(
            tensor=self.tensor,
            active_groups=stage.active_groups,
            model_version=stage_version,
            base_params=self.current_params,
            block_names=block_names,
            dynamics_mode=stage.dynamics_mode,
            frozen_copy_distributions=self._frozen_copy_distributions() if stage.dynamics_mode == "state_only" else None,
            empirical_abundance_proxy=self._empirical_abundance_proxy() if stage.dynamics_mode == "ecDNA_only" else None,
        )
        if objective.adapter.dimension == 0:
            return V4LiteStageFitResult(stage.name, stage.active_groups, block_names, None, None, (), skipped_reason="No active v4-lite parameters for this stage.")
        initial_vector = objective.adapter.default_vector()
        before = objective.evaluate_vector(initial_vector).total_objective
        best_vector, best_value = self._optimize_objective(objective, initial_vector, self.settings.n_restarts)
        evaluated = objective.evaluate_vector(best_vector, return_artifacts=True)
        cfg.require(evaluated.artifacts is not None, "v4-lite stage optimization must return artifacts.")
        posterior_predictive = run_v4_lite_posterior_predictive(objective.observed_summary, evaluated.artifacts.prediction)
        profile = run_v4_lite_profile_likelihood(
            objective,
            best_vector,
            profile_scales=np.maximum(objective.adapter.proposal_scales(), 1e-6),
            n_points=self.settings.profile_points,
            max_dimensions=self.settings.max_profile_dimensions,
            profile_maxiter=self.settings.profile_maxiter,
        )
        fake_data_recovery = run_v4_lite_fake_data_recovery(objective, best_vector, self._optimize_objective, n_restarts=self.settings.fake_recovery_restarts)
        diagnostics = {"posterior_predictive": posterior_predictive, "profile": profile, "fake_data_recovery": fake_data_recovery}
        accepted, rejection_reasons = self._assess_stage(float(before), float(best_value), posterior_predictive, profile, fake_data_recovery, optional=stage.optional)
        return V4LiteStageFitResult(
            stage.name,
            stage.active_groups,
            block_names,
            float(before),
            float(best_value),
            objective.adapter.parameter_names(),
            diagnostics=diagnostics,
            accepted=accepted,
            rejection_reasons=rejection_reasons,
            best_vector=best_vector,
            best_params=evaluated.artifacts.params.copy(),
        )

    def _initial_totals(self) -> dict[str, float]:
        return {condition_name: float(np.sum(values)) for condition_name, values in self.tensor.initial_state_abundance.items()}

    def _empirical_abundance_proxy(self) -> dict[str, np.ndarray]:
        proxy: dict[str, np.ndarray] = {}
        week_to_index = self.tensor.week_to_index
        for condition_name in self.tensor.condition_names:
            values = np.zeros((len(self.tensor.weeks), cfg.N_STATES), dtype=float)
            values[0, :] = self.tensor.initial_state_abundance[condition_name]
            total_by_week: dict[int, float] = {
                observation.week: float(observation.value)
                for observation in self.tensor.count_observations
                if observation.condition == condition_name
            }
            flow_by_week: dict[int, list[np.ndarray]] = {}
            for observation in self.tensor.flow_observations:
                if observation.condition != condition_name or observation.total <= 0.0:
                    continue
                flow_by_week.setdefault(observation.week, []).append(observation.counts / observation.total)
            for week in self.tensor.weeks[1:]:
                previous = values[week_to_index[week] - 1, :]
                fractions = _normalize_simplex(previous)
                if week in flow_by_week:
                    fractions = _normalize_simplex(np.mean(np.stack(flow_by_week[week], axis=0), axis=0))
                total = total_by_week.get(week, float(np.sum(previous)))
                values[week_to_index[week], :] = max(total, 1e-12) * fractions
            proxy[condition_name] = values
        return proxy

    def _frozen_copy_distributions(self) -> dict[str, np.ndarray]:
        if self.ecDNA_reference_prediction is not None:
            return {condition: values.copy() for condition, values in self.ecDNA_reference_prediction.copy_distributions.items()}
        prediction = predict_v4_lite(
            self.tensor,
            self.current_params,
            model_version=self.model_version,
            dynamics_mode="ecDNA_only",
            empirical_abundance_proxy=self._empirical_abundance_proxy(),
        )
        return {condition: values.copy() for condition, values in prediction.copy_distributions.items()}

    def _run_prior_predictive(self) -> V4LitePriorPredictiveReport:
        objective = V4LiteObjective(
            tensor=self.tensor,
            active_groups=("exposure", "ecDNA_kernel", "state_abundance", "burden", "co_segregation"),
            model_version=self.model_version,
            base_params=self.current_params,
        )
        report = run_v4_lite_prior_predictive(objective, n_draws=self.settings.prior_predictive_draws, seed=self.settings.prior_predictive_seed)
        cfg.require(
            report.pass_rate >= self.settings.min_prior_predictive_pass_rate,
            f"v4-lite prior predictive pass rate {report.pass_rate:.3f} is below {self.settings.min_prior_predictive_pass_rate:.3f}.",
        )
        return report

    def _available_blocks(self, requested: tuple[str, ...] | None) -> tuple[str, ...]:
        observed = set(self.tensor.observed_summary.block_names())
        if requested is None:
            return tuple(sorted(observed))
        return tuple(block_name for block_name in requested if block_name in observed)

    def _assess_stage(
        self,
        before: float,
        after: float,
        posterior_predictive: V4LitePosteriorPredictiveReport,
        profile: tuple[V4LiteProfilePoint, ...],
        fake_data_recovery: V4LiteFakeDataRecoveryReport,
        *,
        optional: bool,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        if not np.isfinite(after):
            reasons.append("objective is not finite")
        elif (before - after) < -self.settings.min_objective_improvement:
            reasons.append(f"objective worsened by {after - before:.6g}")
        if posterior_predictive.worst_relative_rmse > self.settings.max_posterior_predictive_relative_rmse:
            reasons.append(f"posterior predictive relative RMSE {posterior_predictive.worst_relative_rmse:.6g} exceeded limit")
        if self.settings.require_fake_data_recovery_pass and not fake_data_recovery.passed:
            reasons.append(f"fake-data recovery failed (normalized_error={fake_data_recovery.normalized_error:.6g})")
        if self._flat_profile_dimensions(profile):
            reasons.append(f"profile likelihood is flat for dimensions {self._flat_profile_dimensions(profile)}")
        if optional and reasons:
            return False, tuple(reasons)
        return (not reasons), tuple(reasons)

    def _flat_profile_dimensions(self, profile: tuple[V4LiteProfilePoint, ...]) -> tuple[int, ...]:
        values_by_dimension: dict[int, list[float]] = {}
        for point in profile:
            values_by_dimension.setdefault(point.dimension_index, []).append(point.objective_value)
        return tuple(sorted(index for index, values in values_by_dimension.items() if max(values) - min(values) < self.settings.min_profile_objective_span))

    def _optimize_objective(self, objective: V4LiteObjective, initial_vector: np.ndarray, n_restarts: int) -> tuple[np.ndarray, float]:
        rng = np.random.default_rng(self.settings.seed)
        proposal_scales = np.maximum(objective.adapter.proposal_scales(), 1e-6)
        starts = [np.asarray(initial_vector, dtype=float).copy()]
        for _ in range(max(0, n_restarts - 1)):
            starts.append(starts[0] + rng.normal(scale=proposal_scales * self.settings.random_start_scale))
        best_vector = starts[0].copy()
        best_value = objective.evaluate_vector(best_vector).total_objective
        for start in starts:
            result = minimize(
                lambda trial: objective.evaluate_vector(trial).total_objective,
                np.asarray(start, dtype=float),
                method="Powell",
                options={"maxiter": self.settings.maxiter, "xtol": 1e-3, "ftol": 1e-3},
            )
            if result.fun < best_value:
                best_vector = np.asarray(result.x, dtype=float).copy()
                best_value = float(result.fun)
        return best_vector, float(best_value)

    def _run_model_comparison(self) -> tuple[dict[str, float], dict[str, dict[str, object]]]:
        scores: dict[str, float] = {}
        fit_results: dict[str, dict[str, object]] = {}
        for version in ("M0", "M1", "M2", "M3"):
            if version == "M3" and not self.tensor.has_same_cell_ectag:
                fit_results[version] = {"skipped_reason": "No same-cell multicolor ecTAG readout."}
                continue
            objective = V4LiteObjective(
                tensor=self.tensor,
                active_groups=("exposure", "ecDNA_kernel", "state_abundance", "burden", "co_segregation"),
                model_version=version,
                base_params=self.current_params,
            )
            vector = objective.adapter.default_vector()
            best_vector, best_value = self._optimize_objective(objective, vector, self.settings.model_comparison_restarts)
            scores[version] = float(best_value)
            fit_results[version] = {
                "objective": float(best_value),
                "dimension": int(objective.adapter.dimension),
                "parameter_names": objective.adapter.parameter_names(),
                "accepted": bool(np.isfinite(best_value)),
                "best_vector": best_vector.tolist(),
            }
        return scores, fit_results

    def _purity_sensitivity_report(self, baseline_prediction: V4LitePrediction) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        baseline_metrics = run_v4_lite_posterior_predictive(self.tensor.observed_summary, baseline_prediction)
        for index, purity_matrix in enumerate(self.purity_sensitivity):
            params = self.current_params.copy()
            params.sort_purity_matrix = _normalize_purity_matrix(np.asarray(purity_matrix, dtype=float))
            prediction = predict_v4_lite(self.tensor, params, model_version=self.model_version)
            metrics = run_v4_lite_posterior_predictive(self.tensor.observed_summary, prediction)
            rows.append(
                {
                    "index": index,
                    "worst_relative_rmse": metrics.worst_relative_rmse,
                    "delta_worst_relative_rmse": metrics.worst_relative_rmse - baseline_metrics.worst_relative_rmse,
                    "block_relative_rmse": metrics.block_relative_rmse,
                }
            )
        return tuple(rows)


def run_v4_lite_prior_predictive(objective: V4LiteObjective, *, n_draws: int, seed: int) -> V4LitePriorPredictiveReport:
    rng = np.random.default_rng(seed)
    default_vector = objective.adapter.default_vector()
    scales = np.maximum(objective.adapter.proposal_scales(), 1e-6)
    failures = {"hard_bounds": 0, "population_extinction": 0, "population_explosion": 0, "state_jump": 0, "ectag_tail": 0, "qpcdr_range": 0}
    passes = 0
    for _ in range(max(1, n_draws)):
        evaluation = objective.evaluate_vector(default_vector + rng.normal(scale=scales), return_artifacts=True)
        if evaluation.artifacts is None:
            failures["hard_bounds"] += 1
            continue
        failure = _prior_predictive_failure_state(evaluation.artifacts.prediction)
        if failure is None:
            passes += 1
        else:
            failures[failure] += 1
    return V4LitePriorPredictiveReport(n_draws=max(1, n_draws), pass_rate=float(passes / max(1, n_draws)), failures=failures)


def _prior_predictive_failure_state(prediction: V4LitePrediction) -> str | None:
    for condition_name in prediction.condition_names:
        abundance = prediction.state_abundance[condition_name]
        totals = np.sum(abundance, axis=1)
        if np.any(~np.isfinite(totals)) or float(np.min(totals)) <= 1e-10:
            return "population_extinction"
        if float(np.max(totals)) / max(float(totals[0]), 1e-12) > 1e4:
            return "population_explosion"
        fractions = abundance / np.maximum(totals[:, None], 1e-12)
        if fractions.shape[0] > 1 and float(np.max(np.sum(np.abs(np.diff(fractions, axis=0)), axis=1))) > 1.5:
            return "state_jump"
        if float(np.max(prediction.copy_distributions[condition_name][:, :, :, -1])) > 0.90:
            return "ectag_tail"
    if "qpcdr" in prediction.summary.blocks and float(np.max(np.abs(prediction.summary.blocks["qpcdr"].values))) > 1e3:
        return "qpcdr_range"
    return None


def run_v4_lite_posterior_predictive(observed_summary: SummaryCollection, prediction: V4LitePrediction) -> V4LitePosteriorPredictiveReport:
    predicted = prediction.summary.align_to(observed_summary)
    rmse: dict[str, float] = {}
    relative: dict[str, float] = {}
    max_abs: dict[str, float] = {}
    for block_name in observed_summary.block_names():
        residual = predicted.blocks[block_name].values - observed_summary.blocks[block_name].values
        block_rmse = float(np.sqrt(np.mean(np.square(residual))))
        scale = max(float(np.sqrt(np.mean(np.square(observed_summary.blocks[block_name].values)))), 1e-6)
        rmse[block_name] = block_rmse
        relative[block_name] = float(block_rmse / scale)
        max_abs[block_name] = float(np.max(np.abs(residual)))
    return V4LitePosteriorPredictiveReport(rmse, relative, max_abs, max(relative.values(), default=0.0))


def run_v4_lite_profile_likelihood(
    objective: V4LiteObjective,
    vector: np.ndarray,
    *,
    profile_scales: np.ndarray,
    n_points: int,
    max_dimensions: int,
    profile_maxiter: int = 25,
) -> tuple[V4LiteProfilePoint, ...]:
    cfg.require(n_points >= 3, "profile likelihood requires at least three points.")
    base = np.asarray(vector, dtype=float)
    offsets = np.linspace(-1.0, 1.0, n_points, dtype=float)
    points: list[V4LiteProfilePoint] = []
    for dimension_index in range(min(max_dimensions, base.size)):
        free_indices = np.array([idx for idx in range(base.size) if idx != dimension_index], dtype=int)
        for offset in offsets.tolist():
            fixed_value = base[dimension_index] + offset * profile_scales[dimension_index]

            def packed(free_values: np.ndarray) -> np.ndarray:
                trial = base.copy()
                trial[dimension_index] = fixed_value
                trial[free_indices] = free_values
                return trial

            if free_indices.size:
                result = minimize(
                    lambda free_values: objective.evaluate_vector(packed(free_values)).total_objective,
                    base[free_indices],
                    method="Powell",
                    options={"maxiter": profile_maxiter, "xtol": 1e-3, "ftol": 1e-3},
                )
                value = float(result.fun)
            else:
                value = objective.evaluate_vector(packed(np.zeros(0, dtype=float))).total_objective
            points.append(V4LiteProfilePoint(dimension_index, float(offset), float(value)))
    return tuple(points)


def _synthetic_summary_from_prediction(objective: V4LiteObjective, params: V4LiteParameters, prediction: V4LitePrediction, rng: np.random.Generator) -> SummaryCollection:
    block_maps = _empty_v4_lite_block_maps()
    for observation in objective.tensor.flow_observations:
        probs = objective._flow_probabilities(prediction, params, observation.condition, observation.week)
        sampled = rng.multinomial(int(max(1, round(observation.total))), probs)
        fractions = sampled / max(float(np.sum(sampled)), 1e-12)
        for state_index, state_name in enumerate(cfg.STATE_NAMES):
            key = f"{observation.condition}|week{observation.week}|state={state_name}|rep={observation.replicate_id}"
            block_maps["flow_fraction"][key] = float(fractions[state_index])
            block_maps["flow_count"][key] = float(sampled[state_index])
    for observation in objective.tensor.count_observations:
        abundance = prediction.state_abundance[observation.condition][objective.tensor.week_to_index[observation.week], :]
        if observation.gate_index is None:
            mu = float(np.sum(abundance))
            block_name = "count_total"
            key = f"{observation.condition}|week{observation.week}|rep={observation.replicate_id}"
        else:
            mu = float(np.sum(params.sort_purity_matrix[observation.gate_index, :] * abundance))
            block_name = "count_gate"
            key = (
                f"{observation.condition}|week{observation.week}|gate={cfg.STATE_NAMES[observation.gate_index]}"
                f"|rep={observation.replicate_id}"
            )
        r = max(float(params.count_dispersion if observation.gate_index is None else params.count_gate_dispersion), 1e-6)
        lam = rng.gamma(shape=r, scale=max(mu / r, 1e-12))
        sampled = float(rng.poisson(lam))
        block_maps[block_name][key] = sampled
    for observation in objective.tensor.qpcdr_observations:
        expected = _expected_qpcdr_value(objective.tensor, params, prediction, observation)
        sampled = float(expected + rng.normal(scale=max(float(params.qpcdr_sigma[observation.species_index]), 1e-8)))
        key = (
            f"{observation.condition}|week{observation.week}|state={cfg.STATE_NAMES[observation.gate_index]}"
            f"|species={cfg.SPECIES[observation.species_index]}|batch={objective.tensor.structure.qpcdr_batches[observation.batch_index]}|rep={observation.replicate_id}"
        )
        block_maps["qpcdr"][key] = sampled
    for observation in objective.tensor.ectag_hist_observations:
        probs = _expected_gate_distribution(
            objective.tensor,
            params,
            prediction,
            observation.condition,
            observation.week,
            observation.gate_index,
            observation.species_index,
        )
        concentration = max(float(params.ectag_concentration[observation.species_index]), 1e-6)
        sampled_probs = rng.dirichlet(np.clip(concentration * probs, 1e-6, None))
        sampled = rng.multinomial(int(max(1, np.sum(observation.counts))), sampled_probs)
        observed_probs = sampled / max(float(np.sum(sampled)), 1e-12)
        prefix = f"{observation.condition}|week{observation.week}|state={cfg.STATE_NAMES[observation.gate_index]}|species={cfg.SPECIES[observation.species_index]}|rep={observation.replicate_id}"
        for bin_index, probability in enumerate(observed_probs.tolist()):
            block_maps["ectag_hist"][f"{prefix}|bin={bin_index}"] = float(probability)
        block_maps["ectag_moments"][f"{prefix}|zero_fraction"] = float(observed_probs[0])
        block_maps["ectag_moments"][f"{prefix}|tail_ge_8"] = objective.tensor.structure.binning.tail_probability(observed_probs, 8)
        block_maps["ectag_moments"][f"{prefix}|tail_ge_16"] = objective.tensor.structure.binning.tail_probability(observed_probs, 16)
    for observation in objective.tensor.ectag_corr_observations:
        expected = params.co_segregation_rho if objective.model_version == "M3" else 0.0
        sigma = params.ectag_corr_sigma / np.sqrt(max(1, observation.n_cells - 1))
        sampled = float(np.clip(expected + rng.normal(scale=sigma), -0.999, 0.999))
        key = (
            f"{observation.condition}|week{observation.week}|state={cfg.STATE_NAMES[observation.gate_index]}"
            f"|pair={cfg.SPECIES[observation.species_a]}-{cfg.SPECIES[observation.species_b]}|rep={observation.replicate_id}"
        )
        block_maps["ectag_corr"][key] = sampled
    return SummaryCollection.from_block_maps(block_maps)


def run_v4_lite_fake_data_recovery(objective: V4LiteObjective, truth_vector: np.ndarray, optimizer, *, n_restarts: int) -> V4LiteFakeDataRecoveryReport:
    truth = objective.evaluate_vector(truth_vector, return_artifacts=True)
    cfg.require(truth.artifacts is not None, "Truth evaluation must return artifacts for fake-data recovery.")
    synthetic_summary = _synthetic_summary_from_prediction(objective, truth.artifacts.params, truth.artifacts.prediction, np.random.default_rng(77))
    synthetic_objective = objective.with_synthetic_observed_summary(synthetic_summary)
    recovered_vector, recovered_value = optimizer(synthetic_objective, synthetic_objective.adapter.default_vector(), n_restarts)
    scale = np.maximum(synthetic_objective.adapter.proposal_scales(), 1e-6)
    normalized_error = float(np.linalg.norm((recovered_vector - truth_vector) / scale) / np.sqrt(max(1, truth_vector.size)))
    recovered_eval = synthetic_objective.evaluate_vector(recovered_vector, return_artifacts=True)
    block_relative: dict[str, float] = {}
    if recovered_eval.artifacts is not None:
        block_relative = run_v4_lite_posterior_predictive(synthetic_summary, recovered_eval.artifacts.prediction).block_relative_rmse
    return V4LiteFakeDataRecoveryReport(float(recovered_value), normalized_error, bool(normalized_error <= 1.5), block_relative)


def _finite_difference_gradient(objective: V4LiteObjective, vector: np.ndarray, step: float = 1e-4) -> np.ndarray:
    grad = np.zeros_like(vector, dtype=float)
    for index in range(vector.size):
        delta = np.zeros_like(vector, dtype=float)
        delta[index] = step
        grad[index] = (objective.evaluate_vector(vector + delta).total_objective - objective.evaluate_vector(vector - delta).total_objective) / (2.0 * step)
    return grad


def run_v4_lite_hmc(objective: V4LiteObjective, initial_vector: np.ndarray, settings: V4LiteOptimizationSettings) -> V4LitePosteriorSamples:
    if settings.posterior_draws <= 0:
        return V4LitePosteriorSamples(objective.adapter.parameter_names(), np.zeros((0, initial_vector.size), dtype=float), 0.0, "posterior_draws <= 0")
    if initial_vector.size == 0:
        return V4LitePosteriorSamples((), np.zeros((0, 0), dtype=float), 0.0, "no active parameters")
    if initial_vector.size > settings.max_hmc_dimensions:
        return V4LitePosteriorSamples(objective.adapter.parameter_names(), np.zeros((0, initial_vector.size), dtype=float), 0.0, "dimension exceeds max_hmc_dimensions")
    rng = np.random.default_rng(settings.seed + 991)
    current = np.asarray(initial_vector, dtype=float).copy()
    current_energy = objective.evaluate_vector(current).total_objective
    samples: list[np.ndarray] = []
    accepted = 0
    total_steps = settings.posterior_burnin + settings.posterior_draws
    step_size = 0.015
    leapfrog_steps = 4
    for step_index in range(total_steps):
        momentum = rng.normal(size=current.shape)
        proposal = current.copy()
        proposal_momentum = momentum.copy()
        grad = _finite_difference_gradient(objective, proposal)
        proposal_momentum -= 0.5 * step_size * grad
        for leapfrog_index in range(leapfrog_steps):
            proposal += step_size * proposal_momentum
            grad = _finite_difference_gradient(objective, proposal)
            if leapfrog_index != leapfrog_steps - 1:
                proposal_momentum -= step_size * grad
        proposal_momentum -= 0.5 * step_size * grad
        proposal_momentum = -proposal_momentum
        proposed_energy = objective.evaluate_vector(proposal).total_objective
        current_h = current_energy + 0.5 * float(np.dot(momentum, momentum))
        proposed_h = proposed_energy + 0.5 * float(np.dot(proposal_momentum, proposal_momentum))
        if np.isfinite(proposed_h) and np.log(rng.uniform()) < min(0.0, current_h - proposed_h):
            current = proposal
            current_energy = proposed_energy
            accepted += 1
        if step_index >= settings.posterior_burnin:
            samples.append(current.copy())
    return V4LitePosteriorSamples(objective.adapter.parameter_names(), np.asarray(samples, dtype=float), float(accepted / max(1, total_steps)))


def run_leave_one_week_out(objective: V4LiteObjective, vector: np.ndarray, optimizer, *, n_restarts: int) -> V4LiteLeaveOneWeekOutReport:
    scores: dict[int, float] = {}
    dynamic_weeks = sorted({week for week in objective.tensor.weeks if week > WEEK1})
    for heldout_week in dynamic_weeks:
        train_objective = V4LiteObjective(
            tensor=objective.tensor,
            active_groups=objective.active_groups,
            model_version=objective.model_version,
            base_params=objective.base_params,
            block_names=objective.block_names,
            synthetic_observed_summary=objective.synthetic_observed_summary,
            dynamics_mode=objective.dynamics_mode,
            heldout_weeks=(heldout_week,),
            frozen_copy_distributions=objective.frozen_copy_distributions,
            empirical_abundance_proxy=objective.empirical_abundance_proxy,
        )
        train_vector, _value = optimizer(train_objective, vector, n_restarts)
        evaluation = train_objective.evaluate_vector(train_vector, return_artifacts=True)
        if evaluation.artifacts is None:
            continue
        predicted = evaluation.artifacts.prediction.summary
        residuals: list[float] = []
        for block_name in objective.observed_summary.block_names():
            observed_block = objective.observed_summary.blocks[block_name]
            predicted_block = predicted.blocks.get(block_name)
            if predicted_block is None:
                continue
            mapping = predicted_block.as_mapping()
            for key, observed_value in observed_block.as_mapping().items():
                if f"|week{heldout_week}" not in key or key not in mapping:
                    continue
                residuals.append(float(mapping[key] - observed_value))
        if residuals:
            scores[heldout_week] = float(np.sqrt(np.mean(np.square(residuals))))
    return V4LiteLeaveOneWeekOutReport(scores)


def run_v4_lite_sbc(runner: V4LiteFitRunner, n_datasets: int, model_version: str) -> V4LiteSBCReport:
    if n_datasets <= 0:
        return V4LiteSBCReport(0, {}, 0, "n_datasets <= 0")
    objective = V4LiteObjective(
        tensor=runner.tensor,
        active_groups=("exposure", "ecDNA_kernel", "state_abundance", "burden", "co_segregation"),
        model_version=model_version,
        base_params=runner.current_params,
    )
    rng = np.random.default_rng(runner.settings.seed + 444)
    default = objective.adapter.default_vector()
    scales = np.maximum(objective.adapter.proposal_scales(), 1e-6)
    rank_map: dict[str, list[int]] = {name: [] for name in objective.adapter.parameter_names()}
    failures = 0
    for _ in range(n_datasets):
        truth_vector = default + rng.normal(scale=scales)
        truth = objective.evaluate_vector(truth_vector, return_artifacts=True)
        if truth.artifacts is None:
            failures += 1
            continue
        synthetic = objective.with_synthetic_observed_summary(_synthetic_summary_from_prediction(objective, truth.artifacts.params, truth.artifacts.prediction, rng))
        recovered, _value = runner._optimize_objective(synthetic, synthetic.adapter.default_vector(), 1)
        if recovered.size <= runner.settings.max_hmc_dimensions:
            recovered_samples = run_v4_lite_hmc(synthetic, recovered, runner.settings).samples
            samples = recovered_samples if recovered_samples.size else recovered + rng.normal(scale=0.5 * scales, size=(25, recovered.size))
        else:
            samples = recovered + rng.normal(scale=0.5 * scales, size=(25, recovered.size))
        ranks = np.sum(samples < truth_vector[None, :], axis=0)
        for name, rank in zip(objective.adapter.parameter_names(), ranks.tolist()):
            rank_map[name].append(int(rank))
    return V4LiteSBCReport(n_datasets, {name: tuple(values) for name, values in rank_map.items()}, failures, "no-NUTS approximate SBC")


def _spec_by_dimension(adapter: V4LiteParameterAdapter) -> tuple[V4LiteFieldSpec, ...]:
    specs: list[V4LiteFieldSpec] = []
    for spec in adapter.specs:
        specs.extend([spec] * spec.unconstrained_size)
    return tuple(specs)


def _boundary_margin_for_spec(spec: V4LiteFieldSpec, raw: np.ndarray) -> float | None:
    if spec.lower is None or spec.upper is None:
        return None
    values = np.asarray(raw, dtype=float).reshape(-1)
    lower = np.asarray(spec.lower, dtype=float).reshape(-1)
    upper = np.asarray(spec.upper, dtype=float).reshape(-1)
    if lower.size != values.size or upper.size != values.size:
        return None
    margins: list[float] = []
    finite_lower = np.isfinite(lower)
    if np.any(finite_lower):
        denom = np.maximum(1.0, np.maximum(np.abs(values[finite_lower]), np.abs(spec.prior_center[finite_lower])))
        margins.extend(((values[finite_lower] - lower[finite_lower]) / denom).tolist())
    finite_upper = np.isfinite(upper)
    if np.any(finite_upper):
        denom = np.maximum(1e-8, upper[finite_upper] - lower[finite_upper])
        margins.extend(((upper[finite_upper] - values[finite_upper]) / denom).tolist())
    finite_margins = [float(value) for value in margins if np.isfinite(value)]
    return min(finite_margins) if finite_margins else None


def build_parameter_status_table(
    objective: V4LiteObjective,
    vector: np.ndarray,
    profile: tuple[V4LiteProfilePoint, ...],
    fake_recovery: V4LiteFakeDataRecoveryReport,
    posterior_samples: V4LitePosteriorSamples | None,
) -> tuple[dict[str, object], ...]:
    profile_span: dict[int, float] = {}
    for point in profile:
        profile_span.setdefault(point.dimension_index, [])
        profile_span[point.dimension_index].append(point.objective_value)
    spans = {index: float(max(values) - min(values)) for index, values in profile_span.items()}
    names = objective.adapter.parameter_names()
    specs_by_dimension = _spec_by_dimension(objective.adapter)
    prior_scales = np.maximum(objective.adapter.proposal_scales(), 1e-8)
    params = objective.adapter.unpack(vector)
    raw_margin_by_spec = {
        spec.name: _boundary_margin_for_spec(spec, objective.adapter._raw(params, spec))
        for spec in objective.adapter.specs
    }
    rows: list[dict[str, object]] = []
    for index, name in enumerate(names):
        span = spans.get(index, float("nan"))
        spec = specs_by_dimension[index] if index < len(specs_by_dimension) else None
        boundary_margin = None if spec is None else raw_margin_by_spec.get(spec.name)
        status = "free"
        rationale_parts: list[str] = []
        if np.isfinite(span) and span < 1e-5:
            status = "fixed"
            rationale_parts.append("flat profile")
        if not fake_recovery.passed:
            status = "derived"
            rationale_parts.append("fake-data recovery did not pass")
        posterior_sd = None
        if posterior_samples is not None and posterior_samples.samples.size and index < posterior_samples.samples.shape[1]:
            posterior_sd = float(np.std(posterior_samples.samples[:, index]))
            if posterior_sd >= prior_scales[index]:
                status = "fixed" if status == "free" else status
                rationale_parts.append("posterior spread not narrower than prior scale")
        elif posterior_samples is not None and posterior_samples.skipped_reason:
            rationale_parts.append(f"posterior_not_evaluated: {posterior_samples.skipped_reason}")
            status = "fixed" if status == "free" else status
        elif posterior_samples is None:
            rationale_parts.append("posterior_not_evaluated")
            status = "fixed" if status == "free" else status
        if boundary_margin is not None and boundary_margin < 0.02:
            status = "fixed" if status == "free" else status
            rationale_parts.append("boundary warning: parameter near hard/soft bound")
        if not rationale_parts:
            rationale_parts.append("profile/fake-data/posterior checks passed")
        rows.append(
            {
                "name": name,
                "field": None if spec is None else spec.name,
                "transform": None if spec is None else spec.transform,
                "prior_kind": None
                if spec is None
                else ("gaussian_shrinkage_approximation" if spec.shrinkage else "gaussian"),
                "fake_data_passed": bool(fake_recovery.passed),
                "status": status,
                "profile_span": span,
                "posterior_sd": posterior_sd,
                "prior_scale": float(prior_scales[index]) if index < prior_scales.size else None,
                "boundary_margin": boundary_margin,
                "rationale": "; ".join(rationale_parts),
            }
        )
    return tuple(rows)


def build_prior_diagnostics_report(objective: V4LiteObjective, vector: np.ndarray) -> dict[str, object]:
    params = objective.adapter.unpack(vector)
    fields: list[dict[str, object]] = []
    for spec in objective.adapter.specs:
        raw = objective.adapter._raw(params, spec)
        fields.append(
            {
                "name": spec.name,
                "group": spec.group,
                "transform": spec.transform,
                "prior_center": spec.prior_center.tolist(),
                "prior_scale": spec.prior_scale.tolist(),
                "prior_kind": "gaussian_shrinkage_approximation" if spec.shrinkage else "gaussian",
                "shrinkage": bool(spec.shrinkage),
                "boundary_type": spec.boundary_type,
                "lower": None if spec.lower is None else spec.lower.tolist(),
                "upper": None if spec.upper is None else spec.upper.tolist(),
                "boundary_margin": _boundary_margin_for_spec(spec, raw),
                "strict_horseshoe_or_pc_prior": "not_implemented_strictly" if spec.shrinkage else "not_applicable",
            }
        )
    return {
        "active_parameter_count": objective.adapter.dimension,
        "active_fields": fields,
        "sampling_note": "posterior samples use the bundled simplified HMC when enabled; this is not NUTS.",
        "strict_horseshoe_prior": "not_implemented_strictly",
        "strict_pc_prior": "not_implemented_strictly",
    }


def _posterior_predictive_residual_rows(observed: SummaryCollection, predicted: SummaryCollection) -> tuple[dict[str, object], ...]:
    aligned = predicted.align_to(observed)
    rows: list[dict[str, object]] = []
    for block_name in observed.block_names():
        observed_block = observed.blocks[block_name]
        predicted_block = aligned.blocks[block_name]
        for key, observed_value, predicted_value in zip(observed_block.keys, observed_block.values.tolist(), predicted_block.values.tolist()):
            rows.append(
                {
                    "block": block_name,
                    "key": key,
                    "observed": float(observed_value),
                    "predicted": float(predicted_value),
                    "residual": float(predicted_value - observed_value),
                }
            )
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
) -> V4LiteReports:
    posterior_metrics = run_v4_lite_posterior_predictive(tensor.observed_summary, prediction)
    residual_rows = _posterior_predictive_residual_rows(tensor.observed_summary, prediction.summary)
    observation_stage = next((stage for stage in stage_results if stage.stage_name == "observation"), None)
    calibration_payload = calibration_report or (observation_stage.diagnostics.get("calibration", {}) if observation_stage is not None else {})
    tensor_summary = {
        "conditions": tensor.condition_names,
        "weeks": tensor.weeks,
        "n_flow": len(tensor.flow_observations),
        "n_count": len(tensor.count_observations),
        "n_count_total": sum(1 for observation in tensor.count_observations if observation.gate_index is None),
        "n_count_gate": sum(1 for observation in tensor.count_observations if observation.gate_index is not None),
        "n_qpcdr": len(tensor.qpcdr_observations),
        "n_ectag_hist": len(tensor.ectag_hist_observations),
        "n_ectag_corr": len(tensor.ectag_corr_observations),
        "binning": {"bins": tensor.structure.binning.bins, "centers": tensor.structure.binning.centers.tolist()},
        "exposure_C": {condition: values.tolist() for condition, values in tensor.exposure_C.items()},
        "exposure_P": {condition: values.tolist() for condition, values in tensor.exposure_P.items()},
        "week1_initial_state_abundance": {condition: values.tolist() for condition, values in tensor.initial_state_abundance.items()},
    }
    calibration_summary = {
        "qpcdr_batches": tensor.structure.qpcdr_batches,
        "has_purity_matrix": tensor.dataset.purity_matrix is not None,
        "has_same_cell_ectag": tensor.has_same_cell_ectag,
        "observation_calibration": calibration_payload,
    }
    ecDNA_report = {
        "tail_max": float(max(np.max(values[:, :, :, -1]) for values in prediction.copy_distributions.values())),
        "mean_copy_by_condition": {
            condition: _copy_means(values[-1], tensor.structure.binning).tolist()
            for condition, values in prediction.copy_distributions.items()
        },
    }
    identifiability_report = {
        "parameter_status": list(parameter_status_table),
        "stage_acceptance": {stage.stage_name: stage.accepted for stage in stage_results},
        "stage_rejections": {stage.stage_name: stage.rejection_reasons for stage in stage_results if stage.rejection_reasons},
        "model_comparison": model_comparison,
    }
    posterior_predictive_report = {
        "leave_one_week_out": loo.heldout_scores,
        "available_blocks": tensor.observed_summary.block_names(),
        "block_rmse": posterior_metrics.block_rmse,
        "block_relative_rmse": posterior_metrics.block_relative_rmse,
        "block_max_abs_residual": posterior_metrics.block_max_abs_residual,
        "worst_relative_rmse": posterior_metrics.worst_relative_rmse,
        "purity_sensitivity": purity_sensitivity,
    }
    fake_data_report = {
        "passed": fake_recovery.passed,
        "normalized_error": fake_recovery.normalized_error,
        "recovered_objective": fake_recovery.recovered_objective,
        "block_relative_rmse": fake_recovery.block_relative_rmse,
    }
    count_gate_counts: dict[str, int] = {}
    for observation in tensor.count_observations:
        if observation.gate_index is not None:
            state_name = cfg.STATE_NAMES[observation.gate_index]
            count_gate_counts[state_name] = count_gate_counts.get(state_name, 0) + 1
    count_observation_report = {
        "total_count_observations": sum(1 for observation in tensor.count_observations if observation.gate_index is None),
        "gate_count_observations": sum(1 for observation in tensor.count_observations if observation.gate_index is not None),
        "gate_counts_by_state": count_gate_counts,
        "likelihoods": {
            "count_total": "NegBin(sum_s N[w,s], phi_N)",
            "count_gate": "NegBin(sum_s Pi[g,s] N[w,s], phi_N_gate)",
        },
        "backward_compatibility": "records without gate remain count_total",
    }
    implementation_status_report = {
        "v4_lite": "implemented",
        "count_gate_observation": "implemented",
        "parameter_status_rules": "implemented_with_profile_fake_data_hmc_or_skip_boundary_margin",
        "posterior_sampling": "approximate_simplified_hmc_not_nuts",
        "sbc": "approximate_no_nuts_when_enabled",
        "strict_horseshoe_prior": "not_implemented_strictly",
        "strict_pc_prior": "not_implemented_strictly",
        "full_v4_formal_bayesian_posterior": "not_enabled_extension",
    }
    sbc_payload = None if sbc_report is None else {"n_datasets": sbc_report.n_datasets, "failures": sbc_report.failures, "ranks": sbc_report.ranks, "skipped_reason": sbc_report.skipped_reason}
    return V4LiteReports(
        tensor_summary=tensor_summary,
        calibration_report=calibration_summary,
        ecDNA_report=ecDNA_report,
        identifiability_report=identifiability_report,
        posterior_predictive_report=posterior_predictive_report,
        fake_data_report=fake_data_report,
        implementation_status_report=implementation_status_report,
        prior_diagnostics_report={} if prior_diagnostics_report is None else prior_diagnostics_report,
        count_observation_report=count_observation_report,
        posterior_predictive_residuals=residual_rows,
        sbc_report=sbc_payload,
    )


def write_v4_lite_reports(
    output_dir: Path,
    reports: V4LiteReports,
    parameter_status_table: tuple[dict[str, object], ...],
    model_comparison: dict[str, float],
    *,
    write_optional_plots: bool = True,
) -> None:
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
    }
    (output_dir / "v4_lite_reports.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (output_dir / "parameter_status.json").write_text(json.dumps(list(parameter_status_table), indent=2, sort_keys=True, default=str), encoding="utf-8")
    (output_dir / "cleaned_tensor_summary.json").write_text(json.dumps(reports.tensor_summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (output_dir / "observation_calibration_report.json").write_text(json.dumps(reports.calibration_report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (output_dir / "ecDNA_only_report.json").write_text(json.dumps(reports.ecDNA_report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (output_dir / "identifiability_report.json").write_text(json.dumps(reports.identifiability_report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (output_dir / "posterior_predictive_report.json").write_text(json.dumps(reports.posterior_predictive_report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (output_dir / "count_observation_report.json").write_text(json.dumps(reports.count_observation_report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (output_dir / "prior_diagnostics_report.json").write_text(json.dumps(reports.prior_diagnostics_report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (output_dir / "implementation_status_report.json").write_text(json.dumps(reports.implementation_status_report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    if reports.sbc_report is not None:
        (output_dir / "sbc_report.json").write_text(json.dumps(reports.sbc_report, indent=2, sort_keys=True, default=str), encoding="utf-8")

    with open(output_dir / "parameter_status.csv", "w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "name",
            "field",
            "transform",
            "prior_kind",
            "fake_data_passed",
            "status",
            "profile_span",
            "posterior_sd",
            "prior_scale",
            "boundary_margin",
            "rationale",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in parameter_status_table:
            writer.writerow({name: row.get(name) for name in fieldnames})

    with open(output_dir / "posterior_predictive_residuals.csv", "w", encoding="utf-8", newline="") as handle:
        fieldnames = ("block", "key", "observed", "predicted", "residual")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in reports.posterior_predictive_residuals:
            writer.writerow({name: row.get(name) for name in fieldnames})

    if write_optional_plots:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return
        blocks = reports.posterior_predictive_report.get("block_relative_rmse", {})
        if isinstance(blocks, dict) and blocks:
            labels = list(blocks)
            values = [float(blocks[label]) for label in labels]
            fig, ax = plt.subplots(figsize=(max(4.0, 0.7 * len(labels)), 3.0))
            ax.bar(labels, values)
            ax.set_ylabel("relative RMSE")
            ax.set_title("v4-lite posterior predictive")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(output_dir / "posterior_predictive_relative_rmse.png", dpi=150)
            plt.close(fig)


def _projection_from_prediction(prediction: V4LitePrediction) -> FullToLiteProjection:
    condition_name = prediction.condition_names[0]
    return FullToLiteProjection(
        weeks=prediction.weeks,
        state_abundance=prediction.state_abundance[condition_name].copy(),
        copy_distributions=prediction.copy_distributions[condition_name].copy(),
        transition_matrices=prediction.transition_matrices[condition_name].copy(),
        growth_rates=prediction.growth_rates[condition_name].copy(),
        copy_kernels=prediction.copy_kernels[condition_name].copy(),
        diagnostics={"source": "v4-lite prediction coarse targets", "condition": condition_name},
    )


def _event_interval_index(times: np.ndarray, event_time: float) -> int | None:
    interval_index = int(np.searchsorted(times, float(event_time), side="left") - 1)
    if 0 <= interval_index < times.size - 1:
        return interval_index
    return None


def _event_soft_state(payload: Mapping[str, object]) -> np.ndarray | None:
    if "soft_state" not in payload:
        return None
    values = np.asarray(payload["soft_state"], dtype=float)
    if values.shape != (cfg.N_STATES,) or not np.all(np.isfinite(values)):
        return None
    return _normalize_simplex(values)


def _event_copy_numbers(payload: Mapping[str, object]) -> np.ndarray | None:
    if "copy_numbers" not in payload:
        return None
    values = np.asarray(payload["copy_numbers"], dtype=int)
    if values.shape != (cfg.N_SPECIES,):
        return None
    return np.clip(values, 0, None)


def _estimate_event_dynamics(
    simulation_result,
    times: np.ndarray,
    state_abundance: np.ndarray,
    structure: V4LiteStructure,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, object]]:
    events = tuple(getattr(simulation_result, "events", ()) or ())
    diagnostics: dict[str, object] = {"lineage_available": bool(events), "event_count": len(events)}
    if not events or times.size < 2:
        diagnostics["lineage_note"] = "No lineage events available; only N^F and p^F were projected."
        return None, None, None, diagnostics

    n_intervals = times.size - 1
    n_bins = structure.binning.n_bins
    transition_counts = np.zeros((n_intervals, cfg.N_STATES, cfg.N_STATES), dtype=float)
    copy_counts = np.zeros((n_intervals, cfg.N_STATES, cfg.N_SPECIES, n_bins, n_bins), dtype=float)
    used_events = 0

    for event_time, event_type, _cell_id, details in events:
        interval_index = _event_interval_index(times, float(event_time))
        if interval_index is None or not isinstance(details, Mapping):
            continue
        state_pre = details.get("state_pre")
        if not isinstance(state_pre, Mapping):
            continue
        source_weights = _event_soft_state(state_pre)
        source_copies = _event_copy_numbers(state_pre)
        if source_weights is None:
            continue

        post_payloads: list[Mapping[str, object]] = []
        if event_type == "division":
            for key in ("daughter_one", "daughter_two"):
                daughter = details.get(key)
                if isinstance(daughter, Mapping):
                    post_payloads.append(daughter)
        else:
            state_post = details.get("state_post")
            if isinstance(state_post, Mapping):
                post_payloads.append(state_post)

        if not post_payloads:
            continue
        for post_payload in post_payloads:
            target_weights = _event_soft_state(post_payload)
            if target_weights is None:
                continue
            transition_counts[interval_index, :, :] += np.outer(source_weights, target_weights)
            target_copies = _event_copy_numbers(post_payload)
            if source_copies is not None and target_copies is not None:
                for source_state, source_weight in enumerate(source_weights.tolist()):
                    if source_weight <= 0.0:
                        continue
                    for species_index in range(cfg.N_SPECIES):
                        source_bin = structure.binning.bin_index(int(source_copies[species_index]))
                        target_bin = structure.binning.bin_index(int(target_copies[species_index]))
                        copy_counts[interval_index, source_state, species_index, source_bin, target_bin] += source_weight
            used_events += 1

    transition_matrices = np.zeros((n_intervals, cfg.N_STATES, cfg.N_STATES), dtype=float)
    copy_kernels = np.zeros((n_intervals, cfg.N_STATES, cfg.N_SPECIES, n_bins, n_bins), dtype=float)
    for interval_index in range(n_intervals):
        for source_state in range(cfg.N_STATES):
            row = transition_counts[interval_index, source_state, :].copy()
            if float(np.sum(row)) <= 0.0:
                row[source_state] = 1.0
            transition_matrices[interval_index, source_state, :] = _normalize_simplex(row)
            for species_index in range(cfg.N_SPECIES):
                for source_bin in range(n_bins):
                    kernel_row = copy_counts[interval_index, source_state, species_index, source_bin, :].copy()
                    if float(np.sum(kernel_row)) <= 0.0:
                        kernel_row[source_bin] = 1.0
                    copy_kernels[interval_index, source_state, species_index, source_bin, :] = _normalize_simplex(kernel_row)

    current = np.maximum(state_abundance[:-1, :], 1e-12)
    nxt = np.maximum(state_abundance[1:, :], 1e-12)
    growth_rates = np.log(nxt / current)
    diagnostics["used_transition_events"] = used_events
    diagnostics["lineage_note"] = "T^F and G^F estimated from recorded event payloads; g^F is net state growth between snapshots."
    return transition_matrices, growth_rates, copy_kernels, diagnostics


def project_full_to_lite(
    simulation_result,
    structure: V4LiteStructure | None = None,
    purity_matrix: np.ndarray | None = None,
) -> FullToLiteProjection:
    model_structure = V4LiteStructure.default() if structure is None else structure
    weeks = tuple(int(round(float(time))) + 1 for time in simulation_result.times)
    n_weeks = len(weeks)
    n_bins = model_structure.binning.n_bins
    state_abundance = np.zeros((n_weeks, cfg.N_STATES), dtype=float)
    copy_distributions = np.zeros((n_weeks, cfg.N_STATES, cfg.N_SPECIES, n_bins), dtype=float)
    has_cell_snapshots = bool(getattr(simulation_result, "cell_snapshots", None))
    for week_index in range(n_weeks):
        cells = simulation_result.cell_snapshots[week_index] if has_cell_snapshots and week_index < len(simulation_result.cell_snapshots) else []
        if cells:
            soft_states = np.asarray([cell["soft_state"] for cell in cells], dtype=float)
            copies = np.asarray([cell["copy_numbers"] for cell in cells], dtype=int)
            state_abundance[week_index, :] = np.sum(soft_states, axis=0)
            for state_index in range(cfg.N_STATES):
                weights = soft_states[:, state_index]
                total = float(np.sum(weights))
                for species_index in range(cfg.N_SPECIES):
                    counts = np.zeros(n_bins, dtype=float)
                    for value, weight in zip(copies[:, species_index], weights):
                        counts[model_structure.binning.bin_index(int(value))] += float(weight)
                    if total <= 0.0:
                        counts[0] = 1.0
                    copy_distributions[week_index, state_index, species_index, :] = _normalize_simplex(counts)
        else:
            snapshot = simulation_result.truth_snapshots[week_index]
            population_size = float(snapshot.get("population_size", simulation_result.population_sizes[week_index]))
            fractions = np.asarray(snapshot.get("soft_state_fractions", simulation_result.soft_state_fractions[week_index]), dtype=float)
            state_abundance[week_index, :] = population_size * fractions
            for state_index in range(cfg.N_STATES):
                for species_index in range(cfg.N_SPECIES):
                    copy_distributions[week_index, state_index, species_index, 0] = 1.0
    transition_matrices, growth_rates, copy_kernels, diagnostics = _estimate_event_dynamics(
        simulation_result,
        np.asarray(simulation_result.times, dtype=float),
        state_abundance,
        model_structure,
    )
    diagnostics["cell_snapshots_used"] = has_cell_snapshots
    if purity_matrix is not None:
        diagnostics["purity_matrix_applied"] = True
        diagnostics["purity_matrix"] = _normalize_purity_matrix(purity_matrix).tolist()
    return FullToLiteProjection(weeks, state_abundance, copy_distributions, transition_matrices, growth_rates, copy_kernels, diagnostics)
