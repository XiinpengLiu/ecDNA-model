"""
Week-level v4-lite fitting path.

This module implements the first fitting target described in
``markdown/fit_method.md`` without calling the full agent-based simulator.
The fitted latent objects are weekly state abundance and binned ecDNA
copy-number distributions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import minimize

import config as cfg
from fit.data import CanonicalFitDataset, WEEK1
from fit.summary import SummaryBlock, SummaryCollection


INVALID_OBJECTIVE = 1e18
DEFAULT_COPY_BINS = ((0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, None))
DEFAULT_COPY_BIN_CENTERS = (0.0, 1.0, 2.5, 5.5, 11.5, 24.0)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=float) - float(np.max(values))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    cfg.require(np.isfinite(total) and total > 0.0, "Softmax weights must have a positive finite sum.")
    return weights / total


def _safe_log1p(value: float | np.ndarray) -> float | np.ndarray:
    return np.log1p(np.clip(value, 0.0, None))


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
            if upper is None:
                if integer_value >= lower:
                    return index
            elif lower <= integer_value <= upper:
                return index
        return self.n_bins - 1

    def probabilities(self, values: Iterable[int | float]) -> np.ndarray:
        counts = np.zeros(self.n_bins, dtype=float)
        n_values = 0
        for value in values:
            counts[self.bin_index(value)] += 1.0
            n_values += 1
        if n_values == 0:
            counts[0] = 1.0
            return counts
        return counts / float(n_values)

    def mean(self, probabilities: np.ndarray) -> float:
        probs = np.asarray(probabilities, dtype=float).reshape(self.n_bins)
        return float(np.dot(probs, self.centers))

    def tail_probability(self, probabilities: np.ndarray, threshold: int) -> float:
        probs = np.asarray(probabilities, dtype=float).reshape(self.n_bins)
        total = 0.0
        for index, (lower, upper) in enumerate(self.bins):
            if upper is None or upper >= threshold:
                if lower >= threshold:
                    total += float(probs[index])
                elif upper is not None and lower < threshold <= upper:
                    total += float(probs[index])
        return total


@dataclass(frozen=True)
class V4LiteStructure:
    transition_edges: tuple[tuple[int, int], ...]
    binning: CopyNumberBinning = field(default_factory=CopyNumberBinning)

    @classmethod
    def default(cls) -> "V4LiteStructure":
        edges = tuple(sorted(cfg.DEFAULT_MODEL_PARAMETERS.generator.base_edges))
        return cls(transition_edges=edges)

    @property
    def n_edges(self) -> int:
        return len(self.transition_edges)


@dataclass
class V4LiteParameters:
    qpcdr_intercept: np.ndarray = field(default_factory=lambda: np.zeros(cfg.N_SPECIES, dtype=float))
    qpcdr_slope: np.ndarray = field(default_factory=lambda: np.ones(cfg.N_SPECIES, dtype=float))
    qpcdr_sigma: np.ndarray = field(default_factory=lambda: np.full(cfg.N_SPECIES, 0.25, dtype=float))
    flow_fraction_sigma: float = 0.05
    count_log_sigma: float = 0.25
    ectag_hist_sigma: float = 0.05
    ectag_moment_sigma: float = 0.10
    exposure_C_scale: float = 1.0
    exposure_P_scale: float = 1.0
    kernel_up_species: np.ndarray = field(default_factory=lambda: np.full(cfg.N_SPECIES, -2.20, dtype=float))
    kernel_down_species: np.ndarray = field(default_factory=lambda: np.full(cfg.N_SPECIES, -2.30, dtype=float))
    kernel_up_state: np.ndarray = field(default_factory=lambda: np.zeros(cfg.N_STATES, dtype=float))
    kernel_down_state: np.ndarray = field(default_factory=lambda: np.zeros(cfg.N_STATES, dtype=float))
    kernel_up_C: np.ndarray = field(default_factory=lambda: np.zeros(cfg.N_SPECIES, dtype=float))
    kernel_up_P: np.ndarray = field(default_factory=lambda: np.zeros(cfg.N_SPECIES, dtype=float))
    kernel_down_C: np.ndarray = field(default_factory=lambda: np.zeros(cfg.N_SPECIES, dtype=float))
    kernel_down_P: np.ndarray = field(default_factory=lambda: np.zeros(cfg.N_SPECIES, dtype=float))
    growth_base: np.ndarray = field(default_factory=lambda: np.zeros(cfg.N_STATES, dtype=float))
    growth_C: np.ndarray = field(default_factory=lambda: np.zeros(cfg.N_STATES, dtype=float))
    growth_P: np.ndarray = field(default_factory=lambda: np.zeros(cfg.N_STATES, dtype=float))
    transition_intercept: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    transition_copy_effect: np.ndarray = field(default_factory=lambda: np.zeros((0, cfg.N_SPECIES), dtype=float))
    transition_C: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    transition_P: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    sort_purity_matrix: np.ndarray = field(default_factory=lambda: np.eye(cfg.N_STATES, dtype=float))

    @classmethod
    def default(cls, structure: V4LiteStructure | None = None) -> "V4LiteParameters":
        model_structure = V4LiteStructure.default() if structure is None else structure
        params = cls()
        base_edges = cfg.DEFAULT_MODEL_PARAMETERS.generator.base_edges
        params.transition_intercept = np.asarray(
            [np.log(max(float(base_edges.get(edge, 0.05)), 1e-4)) for edge in model_structure.transition_edges],
            dtype=float,
        )
        params.transition_copy_effect = np.zeros((model_structure.n_edges, cfg.N_SPECIES), dtype=float)
        params.transition_C = np.zeros(model_structure.n_edges, dtype=float)
        params.transition_P = np.zeros(model_structure.n_edges, dtype=float)
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
    observed_summary: SummaryCollection


@dataclass(frozen=True)
class V4LitePrediction:
    condition_names: tuple[str, ...]
    weeks: tuple[int, ...]
    state_abundance: dict[str, np.ndarray]
    copy_distributions: dict[str, np.ndarray]
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


def build_v4_lite_tensor(
    dataset: CanonicalFitDataset,
    *,
    condition_names: Iterable[str] | None = None,
    structure: V4LiteStructure | None = None,
) -> V4LiteTensor:
    model_structure = V4LiteStructure.default() if structure is None else structure
    selected_conditions = tuple(dataset.condition_names() if condition_names is None else tuple(condition_names))
    cfg.require(bool(selected_conditions), "At least one condition is required for v4-lite fitting.")
    max_week = max(dataset.dynamic_weeks())
    weeks = tuple(range(WEEK1, int(max_week) + 1))
    observed_summary = summarize_dataset_v4_lite(
        dataset,
        condition_names=selected_conditions,
        binning=model_structure.binning,
        dynamic_only=True,
    )

    initial_state_abundance: dict[str, np.ndarray] = {}
    initial_copy_distributions: dict[str, np.ndarray] = {}
    exposure_C: dict[str, np.ndarray] = {}
    exposure_P: dict[str, np.ndarray] = {}

    for condition_name in selected_conditions:
        cfg.require(condition_name in dataset.conditions, f"Unknown condition {condition_name}.")
        initialization = dataset.build_empirical_initialization(condition_name)
        cfg.require(initialization.empirical_flow_fractions is not None, "Empirical week1 flow fractions are required.")
        cfg.require(initialization.empirical_sorted_copy_distributions is not None, "Empirical week1 copy distributions are required.")
        total_count = _week1_total_count(dataset, dataset.resolve_initialization_condition(condition_name))
        initial_state_abundance[condition_name] = total_count * np.asarray(initialization.empirical_flow_fractions, dtype=float)

        p0 = np.zeros((cfg.N_STATES, cfg.N_SPECIES, model_structure.binning.n_bins), dtype=float)
        for state_index, state_name in enumerate(cfg.STATE_NAMES):
            matrix = np.asarray(initialization.empirical_sorted_copy_distributions[state_name], dtype=int)
            for species_index, _species_name in enumerate(cfg.SPECIES):
                p0[state_index, species_index, :] = model_structure.binning.probabilities(matrix[:, species_index])
        initial_copy_distributions[condition_name] = p0

        schedules = dataset.conditions[condition_name].build_input_schedules()
        c_values: list[float] = []
        p_values: list[float] = []
        for week in weeks[:-1]:
            midpoint = float(week - WEEK1) + 0.5
            c_values.append(float(schedules["u_C"](midpoint)))
            p_values.append(float(schedules["u_P"](midpoint)))
        exposure_C[condition_name] = np.asarray(c_values, dtype=float)
        exposure_P[condition_name] = np.asarray(p_values, dtype=float)

    return V4LiteTensor(
        dataset=dataset,
        structure=model_structure,
        condition_names=selected_conditions,
        weeks=weeks,
        initial_state_abundance=initial_state_abundance,
        initial_copy_distributions=initial_copy_distributions,
        exposure_C=exposure_C,
        exposure_P=exposure_P,
        observed_summary=observed_summary,
    )


def _week1_total_count(dataset: CanonicalFitDataset, condition_name: str) -> float:
    count_values = [float(record.value) for record in dataset.counts if record.condition == condition_name and record.week == WEEK1]
    if count_values:
        return max(float(np.mean(count_values)), 1e-6)

    flow_rows = [record for record in dataset.flow if record.condition == condition_name and record.week == WEEK1]
    counted = [float(record.count) for record in flow_rows if record.count is not None]
    if counted:
        return max(float(np.sum(counted)), 1e-6)

    total_events = [float(record.total_events) for record in flow_rows if record.total_events is not None]
    if total_events:
        return max(float(np.mean(total_events)), 1e-6)
    return 1.0


def summarize_dataset_v4_lite(
    dataset: CanonicalFitDataset,
    *,
    condition_names: Iterable[str] | None = None,
    binning: CopyNumberBinning | None = None,
    dynamic_only: bool = True,
) -> SummaryCollection:
    selected_conditions = set(dataset.condition_names() if condition_names is None else tuple(condition_names))
    copy_binning = CopyNumberBinning() if binning is None else binning
    block_maps = _empty_v4_lite_block_maps()

    flow_groups: dict[tuple[str, int, str], list[tuple[int | None, float | None, int | None]]] = {}
    for record in dataset.flow:
        if record.condition not in selected_conditions or (dynamic_only and record.week == WEEK1):
            continue
        flow_groups.setdefault((record.condition, record.week, record.state), []).append(
            (record.count, record.fraction, record.total_events)
        )

    mean_counts: dict[tuple[str, int, str], float] = {}
    count_totals: dict[tuple[str, int], float] = {}
    mean_fractions: dict[tuple[str, int, str], float] = {}
    for key, rows in flow_groups.items():
        condition_name, week, state_name = key
        counts = [float(count) for count, _fraction, _total in rows if count is not None]
        fractions = [float(fraction) for _count, fraction, _total in rows if fraction is not None]
        if counts:
            mean_count = float(np.mean(counts))
            mean_counts[key] = mean_count
            count_totals[(condition_name, week)] = count_totals.get((condition_name, week), 0.0) + mean_count
        if fractions:
            mean_fractions[key] = float(np.mean(fractions))
        if counts:
            block_maps["flow_count"][f"{condition_name}|week{week}|state={state_name}"] = float(_safe_log1p(mean_counts[key]))

    for key in flow_groups:
        condition_name, week, state_name = key
        if key not in mean_fractions and key in mean_counts:
            total = count_totals.get((condition_name, week), 0.0)
            if total > 0.0:
                mean_fractions[key] = mean_counts[key] / total
        if key in mean_fractions:
            block_maps["flow_fraction"][f"{condition_name}|week{week}|state={state_name}"] = float(mean_fractions[key])

    count_groups: dict[tuple[str, int], list[float]] = {}
    for record in dataset.counts:
        if record.condition not in selected_conditions or (dynamic_only and record.week == WEEK1):
            continue
        count_groups.setdefault((record.condition, record.week), []).append(float(record.value))
    for (condition_name, week), rows in count_groups.items():
        block_maps["count_total"][f"{condition_name}|week{week}"] = float(_safe_log1p(np.mean(rows)))

    qpcdr_groups: dict[tuple[str, int, str, str], list[float]] = {}
    for record in dataset.qpcdr:
        if record.condition not in selected_conditions or (dynamic_only and record.week == WEEK1):
            continue
        qpcdr_groups.setdefault((record.condition, record.week, record.state, record.species), []).append(float(record.value))
    for (condition_name, week, state_name, species_name), rows in qpcdr_groups.items():
        prefix = f"{condition_name}|week{week}|state={state_name}|species={species_name}"
        mean_value = float(np.mean(rows))
        if dataset.qpcdr_scale() == "copy_number":
            mean_value = float(_safe_log1p(mean_value))
        block_maps["qpcdr"][f"{prefix}|mean"] = mean_value

    ectag_groups: dict[tuple[str, int, str, str], list[int]] = {}
    for record in dataset.ectag:
        if record.condition not in selected_conditions or (dynamic_only and record.week == WEEK1):
            continue
        ectag_groups.setdefault((record.condition, record.week, record.state, record.species), []).append(int(record.value))
    for (condition_name, week, state_name, species_name), rows in ectag_groups.items():
        prefix = f"{condition_name}|week{week}|state={state_name}|species={species_name}"
        probabilities = copy_binning.probabilities(rows)
        for bin_index, probability in enumerate(probabilities.tolist()):
            block_maps["ectag_hist"][f"{prefix}|bin={bin_index}"] = float(probability)
        array = np.asarray(rows, dtype=int)
        block_maps["ectag_moments"][f"{prefix}|zero_fraction"] = float(np.mean(array == 0))
        block_maps["ectag_moments"][f"{prefix}|tail_ge_8"] = float(np.mean(array >= 8))
        block_maps["ectag_moments"][f"{prefix}|tail_ge_16"] = float(np.mean(array >= 16))

    return SummaryCollection.from_block_maps(block_maps)


def _empty_v4_lite_block_maps() -> dict[str, dict[str, float]]:
    return {
        "flow_fraction": {},
        "flow_count": {},
        "count_total": {},
        "qpcdr": {},
        "ectag_hist": {},
        "ectag_moments": {},
    }


def predict_v4_lite(
    tensor: V4LiteTensor,
    params: V4LiteParameters,
    *,
    reference: SummaryCollection | None = None,
) -> V4LitePrediction:
    _validate_v4_lite_parameters(params, tensor.structure)
    state_abundance: dict[str, np.ndarray] = {}
    copy_distributions: dict[str, np.ndarray] = {}

    n_weeks = len(tensor.weeks)
    n_bins = tensor.structure.binning.n_bins
    for condition_name in tensor.condition_names:
        abundance = np.zeros((n_weeks, cfg.N_STATES), dtype=float)
        distributions = np.zeros((n_weeks, cfg.N_STATES, cfg.N_SPECIES, n_bins), dtype=float)
        abundance[0, :] = np.asarray(tensor.initial_state_abundance[condition_name], dtype=float)
        distributions[0, :, :, :] = np.asarray(tensor.initial_copy_distributions[condition_name], dtype=float)

        for interval_index in range(n_weeks - 1):
            exposure_C = params.exposure_C_scale * float(tensor.exposure_C[condition_name][interval_index])
            exposure_P = params.exposure_P_scale * float(tensor.exposure_P[condition_name][interval_index])

            for state_index in range(cfg.N_STATES):
                for species_index in range(cfg.N_SPECIES):
                    kernel = _copy_number_kernel(params, tensor.structure, state_index, species_index, exposure_C, exposure_P)
                    distributions[interval_index + 1, state_index, species_index, :] = (
                        distributions[interval_index, state_index, species_index, :] @ kernel
                    )

            current_means = _copy_means(distributions[interval_index, :, :, :], tensor.structure.binning)
            transition = _state_transition_matrix(params, tensor.structure, current_means, exposure_C, exposure_P)
            growth = np.exp(params.growth_base + params.growth_C * exposure_C + params.growth_P * exposure_P)
            abundance[interval_index + 1, :] = growth * (transition.T @ abundance[interval_index, :])
            abundance[interval_index + 1, :] = np.clip(abundance[interval_index + 1, :], 1e-12, None)

        state_abundance[condition_name] = abundance
        copy_distributions[condition_name] = distributions

    summary = _prediction_summary(tensor, params, state_abundance, copy_distributions)
    if reference is not None:
        summary = summary.align_to(reference)
    return V4LitePrediction(
        condition_names=tensor.condition_names,
        weeks=tensor.weeks,
        state_abundance=state_abundance,
        copy_distributions=copy_distributions,
        summary=summary,
    )


def _copy_number_kernel(
    params: V4LiteParameters,
    structure: V4LiteStructure,
    state_index: int,
    species_index: int,
    exposure_C: float,
    exposure_P: float,
) -> np.ndarray:
    up_logit = (
        params.kernel_up_species[species_index]
        + params.kernel_up_state[state_index]
        + params.kernel_up_C[species_index] * exposure_C
        + params.kernel_up_P[species_index] * exposure_P
    )
    down_logit = (
        params.kernel_down_species[species_index]
        + params.kernel_down_state[state_index]
        + params.kernel_down_C[species_index] * exposure_C
        + params.kernel_down_P[species_index] * exposure_P
    )
    stay_probability, up_probability, down_probability = _softmax(np.array([0.0, up_logit, down_logit], dtype=float))
    kernel = np.zeros((structure.binning.n_bins, structure.binning.n_bins), dtype=float)
    for bin_index in range(structure.binning.n_bins):
        kernel[bin_index, bin_index] += stay_probability
        if bin_index == 0:
            kernel[bin_index, bin_index] += down_probability
        else:
            kernel[bin_index, bin_index - 1] += down_probability
        if bin_index == structure.binning.n_bins - 1:
            kernel[bin_index, bin_index] += up_probability
        else:
            kernel[bin_index, bin_index + 1] += up_probability
    return kernel


def _copy_means(distributions: np.ndarray, binning: CopyNumberBinning) -> np.ndarray:
    means = np.zeros((cfg.N_STATES, cfg.N_SPECIES), dtype=float)
    for state_index in range(cfg.N_STATES):
        for species_index in range(cfg.N_SPECIES):
            means[state_index, species_index] = binning.mean(distributions[state_index, species_index, :])
    return means


def _state_transition_matrix(
    params: V4LiteParameters,
    structure: V4LiteStructure,
    copy_means: np.ndarray,
    exposure_C: float,
    exposure_P: float,
) -> np.ndarray:
    matrix = np.eye(cfg.N_STATES, dtype=float)
    scaled_means = _safe_log1p(copy_means) / max(float(_safe_log1p(structure.binning.centers[-1])), 1e-6)
    for edge_index, (source, target) in enumerate(structure.transition_edges):
        score = (
            params.transition_intercept[edge_index]
            + float(np.dot(params.transition_copy_effect[edge_index, :], scaled_means[source, :]))
            + params.transition_C[edge_index] * exposure_C
            + params.transition_P[edge_index] * exposure_P
        )
        matrix[source, target] = float(np.exp(np.clip(score, -50.0, 50.0)))
    for state_index in range(cfg.N_STATES):
        matrix[state_index, :] /= float(np.sum(matrix[state_index, :]))
    return matrix


def _prediction_summary(
    tensor: V4LiteTensor,
    params: V4LiteParameters,
    state_abundance: Mapping[str, np.ndarray],
    copy_distributions: Mapping[str, np.ndarray],
) -> SummaryCollection:
    block_maps = _empty_v4_lite_block_maps()
    week_to_index = {week: index for index, week in enumerate(tensor.weeks)}
    purity = np.asarray(params.sort_purity_matrix, dtype=float)

    for condition_name in tensor.condition_names:
        abundance = state_abundance[condition_name]
        distributions = copy_distributions[condition_name]
        for week in tensor.weeks:
            if week == WEEK1:
                continue
            week_index = week_to_index[week]
            total = float(np.sum(abundance[week_index, :]))
            fractions = abundance[week_index, :] / max(total, 1e-12)
            observed_fractions = purity @ fractions
            observed_fractions = observed_fractions / max(float(np.sum(observed_fractions)), 1e-12)
            block_maps["count_total"][f"{condition_name}|week{week}"] = float(_safe_log1p(total))
            for state_index, state_name in enumerate(cfg.STATE_NAMES):
                state_prefix = f"{condition_name}|week{week}|state={state_name}"
                block_maps["flow_fraction"][state_prefix] = float(observed_fractions[state_index])
                block_maps["flow_count"][state_prefix] = float(_safe_log1p(abundance[week_index, state_index]))
                for species_index, species_name in enumerate(cfg.SPECIES):
                    prefix = f"{state_prefix}|species={species_name}"
                    probabilities = distributions[week_index, state_index, species_index, :]
                    mean_copy = tensor.structure.binning.mean(probabilities)
                    if tensor.dataset.qpcdr_scale() == "ct":
                        qpcdr_value = params.qpcdr_intercept[species_index] - params.qpcdr_slope[species_index] * np.log10(
                            mean_copy + 1e-6
                        )
                    else:
                        qpcdr_value = params.qpcdr_intercept[species_index] + params.qpcdr_slope[species_index] * float(
                            _safe_log1p(mean_copy)
                        )
                    block_maps["qpcdr"][f"{prefix}|mean"] = float(qpcdr_value)
                    for bin_index, probability in enumerate(probabilities.tolist()):
                        block_maps["ectag_hist"][f"{prefix}|bin={bin_index}"] = float(probability)
                    block_maps["ectag_moments"][f"{prefix}|zero_fraction"] = float(probabilities[0])
                    block_maps["ectag_moments"][f"{prefix}|tail_ge_8"] = tensor.structure.binning.tail_probability(
                        probabilities,
                        8,
                    )
                    block_maps["ectag_moments"][f"{prefix}|tail_ge_16"] = tensor.structure.binning.tail_probability(
                        probabilities,
                        16,
                    )

    return SummaryCollection.from_block_maps(block_maps)


def _validate_v4_lite_parameters(params: V4LiteParameters, structure: V4LiteStructure) -> None:
    for field_name in ("qpcdr_intercept", "qpcdr_slope", "qpcdr_sigma"):
        values = np.asarray(getattr(params, field_name), dtype=float)
        cfg.require(values.shape == (cfg.N_SPECIES,), f"{field_name} must have shape ({cfg.N_SPECIES},).")
        cfg.require(np.all(np.isfinite(values)), f"{field_name} must be finite.")
    for field_name in ("qpcdr_slope", "qpcdr_sigma"):
        cfg.require(np.all(getattr(params, field_name) > 0.0), f"{field_name} must be positive.")
    for field_name in ("flow_fraction_sigma", "count_log_sigma", "ectag_hist_sigma", "ectag_moment_sigma"):
        value = float(getattr(params, field_name))
        cfg.require(np.isfinite(value) and value > 0.0, f"{field_name} must be positive and finite.")
    for field_name in ("exposure_C_scale", "exposure_P_scale"):
        value = float(getattr(params, field_name))
        cfg.require(np.isfinite(value) and value >= 0.0, f"{field_name} must be non-negative and finite.")
    for field_name in (
        "kernel_up_species",
        "kernel_down_species",
        "kernel_up_C",
        "kernel_up_P",
        "kernel_down_C",
        "kernel_down_P",
    ):
        values = np.asarray(getattr(params, field_name), dtype=float)
        cfg.require(values.shape == (cfg.N_SPECIES,), f"{field_name} must have shape ({cfg.N_SPECIES},).")
        cfg.require(np.all(np.isfinite(values)), f"{field_name} must be finite.")
    for field_name in ("kernel_up_state", "kernel_down_state", "growth_base", "growth_C", "growth_P"):
        values = np.asarray(getattr(params, field_name), dtype=float)
        cfg.require(values.shape == (cfg.N_STATES,), f"{field_name} must have shape ({cfg.N_STATES},).")
        cfg.require(np.all(np.isfinite(values)), f"{field_name} must be finite.")
    for field_name in ("transition_intercept", "transition_C", "transition_P"):
        values = np.asarray(getattr(params, field_name), dtype=float)
        cfg.require(values.shape == (structure.n_edges,), f"{field_name} must have shape ({structure.n_edges},).")
        cfg.require(np.all(np.isfinite(values)), f"{field_name} must be finite.")
    cfg.require(
        np.asarray(params.transition_copy_effect, dtype=float).shape == (structure.n_edges, cfg.N_SPECIES),
        f"transition_copy_effect must have shape ({structure.n_edges}, {cfg.N_SPECIES}).",
    )
    purity = np.asarray(params.sort_purity_matrix, dtype=float)
    cfg.require(purity.shape == (cfg.N_STATES, cfg.N_STATES), "sort_purity_matrix has invalid shape.")
    cfg.require(np.all(purity >= 0.0), "sort_purity_matrix must be non-negative.")
    cfg.require(np.allclose(np.sum(purity, axis=0), 1.0, atol=1e-8), "sort_purity_matrix columns must sum to one.")


@dataclass(frozen=True)
class V4LiteFieldSpec:
    name: str
    group: str
    transform: str
    shape: tuple[int, ...]
    prior_center: np.ndarray
    prior_scale: np.ndarray
    shrinkage: bool = False

    @property
    def raw_size(self) -> int:
        return int(np.prod(self.shape, dtype=int)) if self.shape else 1


def _field_specs(structure: V4LiteStructure) -> tuple[V4LiteFieldSpec, ...]:
    def spec(
        name: str,
        group: str,
        transform: str,
        shape: tuple[int, ...],
        center: float | np.ndarray,
        scale: float,
        *,
        shrinkage: bool = False,
    ) -> V4LiteFieldSpec:
        center_array = np.asarray(center, dtype=float)
        if not shape:
            center_array = center_array.reshape(1)
            scale_array = np.full(1, float(scale), dtype=float)
        else:
            center_array = np.broadcast_to(center_array, shape).astype(float).reshape(-1)
            scale_array = np.full(center_array.size, float(scale), dtype=float)
        return V4LiteFieldSpec(
            name=name,
            group=group,
            transform=transform,
            shape=shape,
            prior_center=center_array,
            prior_scale=scale_array,
            shrinkage=shrinkage,
        )

    edge_shape = (structure.n_edges,)
    edge_species_shape = (structure.n_edges, cfg.N_SPECIES)
    return (
        spec("qpcdr_intercept", "observation", "identity", (cfg.N_SPECIES,), 0.0, 0.75),
        spec("qpcdr_slope", "observation", "log", (cfg.N_SPECIES,), 1.0, 0.40),
        spec("qpcdr_sigma", "observation", "log", (cfg.N_SPECIES,), 0.25, 0.35),
        spec("flow_fraction_sigma", "observation", "log", (), 0.05, 0.35),
        spec("count_log_sigma", "observation", "log", (), 0.25, 0.35),
        spec("ectag_hist_sigma", "observation", "log", (), 0.05, 0.35),
        spec("ectag_moment_sigma", "observation", "log", (), 0.10, 0.35),
        spec("exposure_C_scale", "exposure", "log", (), 1.0, 0.50),
        spec("exposure_P_scale", "exposure", "log", (), 1.0, 0.50),
        spec("kernel_up_species", "ecDNA_kernel", "identity", (cfg.N_SPECIES,), -2.20, 0.65),
        spec("kernel_down_species", "ecDNA_kernel", "identity", (cfg.N_SPECIES,), -2.30, 0.65),
        spec("kernel_up_state", "ecDNA_kernel", "identity", (cfg.N_STATES,), 0.0, 0.45, shrinkage=True),
        spec("kernel_down_state", "ecDNA_kernel", "identity", (cfg.N_STATES,), 0.0, 0.45, shrinkage=True),
        spec("kernel_up_C", "ecDNA_kernel", "identity", (cfg.N_SPECIES,), 0.0, 0.45, shrinkage=True),
        spec("kernel_up_P", "ecDNA_kernel", "identity", (cfg.N_SPECIES,), 0.0, 0.45, shrinkage=True),
        spec("kernel_down_C", "ecDNA_kernel", "identity", (cfg.N_SPECIES,), 0.0, 0.45, shrinkage=True),
        spec("kernel_down_P", "ecDNA_kernel", "identity", (cfg.N_SPECIES,), 0.0, 0.45, shrinkage=True),
        spec("growth_base", "state_abundance", "identity", (cfg.N_STATES,), 0.0, 0.40),
        spec("growth_C", "state_abundance", "identity", (cfg.N_STATES,), 0.0, 0.45, shrinkage=True),
        spec("growth_P", "state_abundance", "identity", (cfg.N_STATES,), 0.0, 0.45, shrinkage=True),
        spec("transition_intercept", "state_abundance", "identity", edge_shape, -2.50, 0.80),
        spec("transition_copy_effect", "state_abundance", "identity", edge_species_shape, 0.0, 0.35, shrinkage=True),
        spec("transition_C", "state_abundance", "identity", edge_shape, 0.0, 0.40, shrinkage=True),
        spec("transition_P", "state_abundance", "identity", edge_shape, 0.0, 0.40, shrinkage=True),
    )


class V4LiteParameterAdapter:
    def __init__(
        self,
        *,
        structure: V4LiteStructure,
        base_params: V4LiteParameters,
        active_groups: Iterable[str],
    ):
        self.structure = structure
        self.base_params = base_params.copy()
        groups = set(active_groups)
        self.specs = tuple(spec for spec in _field_specs(structure) if spec.group in groups)
        offset = 0
        slices: list[tuple[int, int]] = []
        for spec in self.specs:
            slices.append((offset, offset + spec.raw_size))
            offset += spec.raw_size
        self.slices = tuple(slices)
        self.dimension = offset

    def default_vector(self) -> np.ndarray:
        return self.pack(self.base_params)

    def pack(self, params: V4LiteParameters) -> np.ndarray:
        pieces: list[np.ndarray] = []
        for spec in self.specs:
            raw = self._raw(params, spec)
            pieces.append(self._to_unconstrained(spec, raw))
        if not pieces:
            return np.zeros(0, dtype=float)
        return np.concatenate(pieces, axis=0)

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
            multiplier = 0.65 if spec.shrinkage else 1.0
            pieces.append(multiplier * spec.prior_scale.copy())
        if not pieces:
            return np.zeros(0, dtype=float)
        return np.concatenate(pieces, axis=0)

    def prior_penalty(self, params: V4LiteParameters) -> float:
        penalty = 0.0
        for spec in self.specs:
            raw = self._raw(params, spec)
            value = self._to_unconstrained(spec, raw)
            center = self._to_unconstrained(spec, spec.prior_center)
            z = (value - center) / np.maximum(spec.prior_scale, 1e-8)
            multiplier = 2.0 if spec.shrinkage else 1.0
            penalty += 0.5 * multiplier * float(np.dot(z, z)) / max(1, z.size)
        return float(penalty)

    @staticmethod
    def _raw(params: V4LiteParameters, spec: V4LiteFieldSpec) -> np.ndarray:
        value = getattr(params, spec.name)
        return np.asarray(value, dtype=float).reshape(-1)

    @staticmethod
    def _to_unconstrained(spec: V4LiteFieldSpec, raw: np.ndarray) -> np.ndarray:
        values = np.asarray(raw, dtype=float).reshape(-1)
        if spec.transform == "identity":
            return values
        if spec.transform == "log":
            return np.log(np.clip(values, 1e-12, None))
        raise ValueError(f"Unsupported v4-lite transform {spec.transform}.")

    @staticmethod
    def _from_unconstrained(spec: V4LiteFieldSpec, values: np.ndarray) -> np.ndarray:
        flat = np.asarray(values, dtype=float).reshape(-1)
        if spec.transform == "identity":
            return flat
        if spec.transform == "log":
            return np.exp(flat)
        raise ValueError(f"Unsupported v4-lite transform {spec.transform}.")


class V4LiteObjective:
    def __init__(
        self,
        *,
        tensor: V4LiteTensor,
        active_groups: Iterable[str],
        base_params: V4LiteParameters | None = None,
        observed_summary: SummaryCollection | None = None,
        block_names: Iterable[str] | None = None,
    ):
        self.tensor = tensor
        self.active_groups = tuple(active_groups)
        self.base_params = V4LiteParameters.default(tensor.structure) if base_params is None else base_params.copy()
        self.observed_summary = tensor.observed_summary if observed_summary is None else observed_summary
        self.block_names = None if block_names is None else set(block_names)
        self.adapter = V4LiteParameterAdapter(
            structure=tensor.structure,
            base_params=self.base_params,
            active_groups=self.active_groups,
        )

    def with_observed_summary(self, observed_summary: SummaryCollection) -> "V4LiteObjective":
        return V4LiteObjective(
            tensor=self.tensor,
            active_groups=self.active_groups,
            base_params=self.base_params,
            observed_summary=observed_summary,
            block_names=self.block_names,
        )

    def evaluate_vector(self, vector: np.ndarray, *, return_artifacts: bool = False) -> V4LiteObjectiveResult:
        try:
            params = self.adapter.unpack(vector)
            prediction = predict_v4_lite(self.tensor, params, reference=self.observed_summary)
        except (ValueError, FloatingPointError):
            return V4LiteObjectiveResult(
                total_objective=INVALID_OBJECTIVE,
                data_negative_log_likelihood=INVALID_OBJECTIVE,
                prior_penalty=0.0,
                block_results=(),
                artifacts=None,
            )

        block_results: list[V4LiteBlockResult] = []
        data_nll = 0.0
        for block_name in self.observed_summary.block_names():
            if self.block_names is not None and block_name not in self.block_names:
                continue
            observed_block = self.observed_summary.blocks[block_name]
            predicted_block = prediction.summary.blocks[block_name]
            block_result = self._block_likelihood(block_name, observed_block, predicted_block, params)
            block_results.append(block_result)
            data_nll += block_result.negative_log_likelihood
        prior_penalty = self.adapter.prior_penalty(params)
        total = data_nll + prior_penalty
        artifacts = None
        if return_artifacts:
            artifacts = V4LiteObjectiveArtifacts(params=params.copy(), prediction=prediction)
        return V4LiteObjectiveResult(
            total_objective=float(total),
            data_negative_log_likelihood=float(data_nll),
            prior_penalty=float(prior_penalty),
            block_results=tuple(block_results),
            artifacts=artifacts,
        )

    def _block_likelihood(
        self,
        block_name: str,
        observed: SummaryBlock,
        predicted: SummaryBlock,
        params: V4LiteParameters,
    ) -> V4LiteBlockResult:
        residual = predicted.values - observed.values
        sigma = _sigma_for_keys(block_name, observed.keys, params)
        normalized = residual / sigma
        nll_terms = 0.5 * (np.square(normalized) + np.log(2.0 * np.pi * np.square(sigma)))
        negative_log_likelihood = float(np.mean(nll_terms))
        residual_norm = float(np.linalg.norm(residual) / np.sqrt(max(1, residual.size)))
        return V4LiteBlockResult(
            name=block_name,
            dimension=int(residual.size),
            negative_log_likelihood=negative_log_likelihood,
            residual_norm=residual_norm,
        )


def _sigma_for_keys(block_name: str, keys: tuple[str, ...], params: V4LiteParameters) -> np.ndarray:
    if block_name == "flow_fraction":
        return np.full(len(keys), params.flow_fraction_sigma, dtype=float)
    if block_name in {"count_total", "flow_count"}:
        return np.full(len(keys), params.count_log_sigma, dtype=float)
    if block_name == "ectag_hist":
        return np.full(len(keys), params.ectag_hist_sigma, dtype=float)
    if block_name == "ectag_moments":
        return np.full(len(keys), params.ectag_moment_sigma, dtype=float)
    if block_name == "qpcdr":
        sigma = np.zeros(len(keys), dtype=float)
        for index, key in enumerate(keys):
            species_name = _species_from_key(key)
            sigma[index] = params.qpcdr_sigma[cfg.SPECIES_INDEX[species_name]]
        return sigma
    return np.ones(len(keys), dtype=float)


def _species_from_key(key: str) -> str:
    marker = "|species="
    cfg.require(marker in key, f"Key {key} does not contain a species token.")
    return key.split(marker, 1)[1].split("|", 1)[0]


@dataclass(frozen=True)
class V4LiteStageDefinition:
    name: str
    active_groups: tuple[str, ...]
    block_names: tuple[str, ...] | None
    description: str


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
    fake_recovery_restarts: int = 2
    min_objective_improvement: float = 1e-7
    max_posterior_predictive_relative_rmse: float = 3.0
    min_profile_objective_span: float = 1e-5
    require_fake_data_recovery_pass: bool = False
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


V4_LITE_STAGE_SEQUENCE = (
    V4LiteStageDefinition(
        name="observation",
        active_groups=("observation",),
        block_names=None,
        description="Assay calibration and block noise.",
    ),
    V4LiteStageDefinition(
        name="ecDNA-only",
        active_groups=("observation", "exposure", "ecDNA_kernel"),
        block_names=("qpcdr", "ectag_hist", "ectag_moments"),
        description="State-specific weekly ecDNA up/down kernel.",
    ),
    V4LiteStageDefinition(
        name="state-only",
        active_groups=("observation", "exposure", "state_abundance"),
        block_names=("flow_fraction", "flow_count", "count_total"),
        description="Weekly net growth and sparse state switching.",
    ),
    V4LiteStageDefinition(
        name="joint-v4-lite",
        active_groups=("observation", "exposure", "ecDNA_kernel", "state_abundance"),
        block_names=None,
        description="Joint v4-lite refinement.",
    ),
)


class V4LiteFitRunner:
    def __init__(
        self,
        dataset: CanonicalFitDataset,
        *,
        structure: V4LiteStructure | None = None,
        initial_params: V4LiteParameters | None = None,
        optimization_settings: V4LiteOptimizationSettings | None = None,
        condition_names: Iterable[str] | None = None,
    ):
        self.structure = V4LiteStructure.default() if structure is None else structure
        self.tensor = build_v4_lite_tensor(dataset, condition_names=condition_names, structure=self.structure)
        self.current_params = V4LiteParameters.default(self.structure) if initial_params is None else initial_params.copy()
        self.settings = V4LiteOptimizationSettings() if optimization_settings is None else optimization_settings

    def run_all_stages(self) -> V4LiteFitResult:
        prior_predictive = self._run_prior_predictive()
        stage_results: list[V4LiteStageFitResult] = []
        for stage in V4_LITE_STAGE_SEQUENCE:
            result = self.run_stage(stage)
            if result.accepted and result.best_params is not None:
                self.current_params = result.best_params.copy()
            stage_results.append(result)
        return V4LiteFitResult(
            prior_predictive=prior_predictive,
            stage_results=stage_results,
            final_params=self.current_params.copy(),
            tensor=self.tensor,
        )

    def run_stage(self, stage: V4LiteStageDefinition) -> V4LiteStageFitResult:
        block_names = self._available_blocks(stage.block_names)
        if not block_names:
            return V4LiteStageFitResult(
                stage_name=stage.name,
                active_groups=stage.active_groups,
                block_names=(),
                objective_before=None,
                objective_after=None,
                active_parameter_names=(),
                skipped_reason="No observed blocks are available for this stage.",
            )

        objective = V4LiteObjective(
            tensor=self.tensor,
            active_groups=stage.active_groups,
            base_params=self.current_params,
            block_names=block_names,
        )
        if objective.adapter.dimension == 0:
            return V4LiteStageFitResult(
                stage_name=stage.name,
                active_groups=stage.active_groups,
                block_names=block_names,
                objective_before=None,
                objective_after=None,
                active_parameter_names=(),
                skipped_reason="No active v4-lite parameters for this stage.",
            )

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
        )
        fake_data_recovery = run_v4_lite_fake_data_recovery(
            objective,
            best_vector,
            self._optimize_objective,
            n_restarts=self.settings.fake_recovery_restarts,
        )
        diagnostics: dict[str, object] = {
            "posterior_predictive": posterior_predictive,
            "profile": profile,
            "fake_data_recovery": fake_data_recovery,
        }
        accepted, rejection_reasons = self._assess_stage(
            objective_before=float(before),
            objective_after=float(best_value),
            posterior_predictive=posterior_predictive,
            profile=profile,
            fake_data_recovery=fake_data_recovery,
        )
        return V4LiteStageFitResult(
            stage_name=stage.name,
            active_groups=stage.active_groups,
            block_names=block_names,
            objective_before=float(before),
            objective_after=float(best_value),
            active_parameter_names=tuple(spec.name for spec in objective.adapter.specs),
            diagnostics=diagnostics,
            accepted=accepted,
            rejection_reasons=rejection_reasons,
            best_vector=best_vector,
            best_params=evaluated.artifacts.params.copy(),
        )

    def _run_prior_predictive(self) -> V4LitePriorPredictiveReport:
        objective = V4LiteObjective(
            tensor=self.tensor,
            active_groups=("observation", "exposure", "ecDNA_kernel", "state_abundance"),
            base_params=self.current_params,
        )
        report = run_v4_lite_prior_predictive(
            objective,
            n_draws=self.settings.prior_predictive_draws,
            seed=self.settings.prior_predictive_seed,
        )
        cfg.require(
            report.pass_rate >= self.settings.min_prior_predictive_pass_rate,
            f"v4-lite prior predictive pass rate {report.pass_rate:.3f} is below "
            f"{self.settings.min_prior_predictive_pass_rate:.3f}.",
        )
        return report

    def _available_blocks(self, requested: tuple[str, ...] | None) -> tuple[str, ...]:
        observed_blocks = set(self.tensor.observed_summary.block_names())
        if requested is None:
            return tuple(sorted(observed_blocks))
        return tuple(block_name for block_name in requested if block_name in observed_blocks)

    def _assess_stage(
        self,
        *,
        objective_before: float,
        objective_after: float,
        posterior_predictive: V4LitePosteriorPredictiveReport,
        profile: tuple[V4LiteProfilePoint, ...],
        fake_data_recovery: V4LiteFakeDataRecoveryReport,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        improvement = objective_before - objective_after
        if not np.isfinite(objective_after):
            reasons.append("objective is not finite")
        elif improvement < -self.settings.min_objective_improvement:
            reasons.append(f"objective worsened by {-improvement:.6g}")
        if not np.isfinite(posterior_predictive.worst_relative_rmse):
            reasons.append("posterior predictive residuals are not finite")
        elif posterior_predictive.worst_relative_rmse > self.settings.max_posterior_predictive_relative_rmse:
            reasons.append(
                "posterior predictive relative RMSE "
                f"{posterior_predictive.worst_relative_rmse:.6g} exceeded "
                f"{self.settings.max_posterior_predictive_relative_rmse:.6g}"
            )
        if self.settings.require_fake_data_recovery_pass and not fake_data_recovery.passed:
            reasons.append(
                "fake-data recovery failed "
                f"(normalized_error={fake_data_recovery.normalized_error:.6g})"
            )
        flat_dimensions = self._flat_profile_dimensions(profile)
        if flat_dimensions:
            reasons.append(f"profile likelihood is flat for dimensions {flat_dimensions}")
        return (not reasons), tuple(reasons)

    def _flat_profile_dimensions(self, profile: tuple[V4LiteProfilePoint, ...]) -> tuple[int, ...]:
        objective_by_dimension: dict[int, list[float]] = {}
        for point in profile:
            objective_by_dimension.setdefault(point.dimension_index, []).append(float(point.objective_value))
        flat = [
            int(dimension)
            for dimension, values in objective_by_dimension.items()
            if max(values) - min(values) < self.settings.min_profile_objective_span
        ]
        return tuple(sorted(flat))

    def _optimize_objective(
        self,
        objective: V4LiteObjective,
        initial_vector: np.ndarray,
        n_restarts: int,
    ) -> tuple[np.ndarray, float]:
        rng = np.random.default_rng(self.settings.seed)
        proposal_scales = np.maximum(objective.adapter.proposal_scales(), 1e-6)
        starts = [np.asarray(initial_vector, dtype=float).copy()]
        for _ in range(max(0, n_restarts - 1)):
            starts.append(starts[0] + rng.normal(scale=proposal_scales * self.settings.random_start_scale))

        best_vector = starts[0].copy()
        best_value = objective.evaluate_vector(best_vector).total_objective
        for start_vector in starts:
            result = minimize(
                lambda trial: objective.evaluate_vector(trial).total_objective,
                np.asarray(start_vector, dtype=float),
                method="Powell",
                options={"maxiter": self.settings.maxiter, "xtol": 1e-3, "ftol": 1e-3},
            )
            if result.fun < best_value:
                best_vector = np.asarray(result.x, dtype=float).copy()
                best_value = float(result.fun)
        return best_vector, float(best_value)


def run_v4_lite_prior_predictive(
    objective: V4LiteObjective,
    *,
    n_draws: int,
    seed: int,
) -> V4LitePriorPredictiveReport:
    cfg.require(n_draws > 0, "n_draws must be positive.")
    rng = np.random.default_rng(seed)
    default_vector = objective.adapter.default_vector()
    proposal_scales = objective.adapter.proposal_scales()
    failures = {
        "hard_bounds": 0,
        "population_extinction": 0,
        "population_explosion": 0,
        "state_jump": 0,
        "ectag_tail": 0,
        "qpcdr_range": 0,
    }
    n_pass = 0
    for _draw in range(n_draws):
        trial = default_vector + rng.normal(scale=proposal_scales, size=proposal_scales.shape)
        result = objective.evaluate_vector(trial, return_artifacts=True)
        if result.artifacts is None:
            failures["hard_bounds"] += 1
            continue
        failure_state = _prior_predictive_failure_state(result.artifacts.prediction)
        if failure_state is None:
            n_pass += 1
        else:
            failures[failure_state] += 1
    return V4LitePriorPredictiveReport(n_draws=n_draws, pass_rate=float(n_pass / n_draws), failures=failures)


def _prior_predictive_failure_state(prediction: V4LitePrediction) -> str | None:
    for condition_name in prediction.condition_names:
        abundance = prediction.state_abundance[condition_name]
        totals = np.sum(abundance, axis=1)
        if np.any(~np.isfinite(totals)) or float(np.min(totals)) <= 1e-10:
            return "population_extinction"
        if float(np.max(totals)) / max(float(totals[0]), 1e-12) > 1e4:
            return "population_explosion"
        fractions = abundance / np.maximum(totals[:, None], 1e-12)
        if fractions.shape[0] >= 2:
            max_jump = float(np.max(np.sum(np.abs(np.diff(fractions, axis=0)), axis=1)))
            if max_jump > 1.5:
                return "state_jump"
        tail = prediction.copy_distributions[condition_name][:, :, :, -1]
        if float(np.max(tail)) > 0.90:
            return "ectag_tail"
    if "qpcdr" in prediction.summary.blocks:
        values = prediction.summary.blocks["qpcdr"].values
        if values.size and float(np.max(np.abs(values))) > 1e3:
            return "qpcdr_range"
    return None


def run_v4_lite_posterior_predictive(
    observed_summary: SummaryCollection,
    prediction: V4LitePrediction,
) -> V4LitePosteriorPredictiveReport:
    predicted = prediction.summary.align_to(observed_summary)
    block_rmse: dict[str, float] = {}
    block_relative_rmse: dict[str, float] = {}
    block_max_abs_residual: dict[str, float] = {}
    for block_name in observed_summary.block_names():
        residual = predicted.blocks[block_name].values - observed_summary.blocks[block_name].values
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        observed_scale = float(np.sqrt(np.mean(np.square(observed_summary.blocks[block_name].values))))
        observed_scale = max(observed_scale, 1e-6)
        block_rmse[block_name] = rmse
        block_relative_rmse[block_name] = float(rmse / observed_scale)
        block_max_abs_residual[block_name] = float(np.max(np.abs(residual)))
    return V4LitePosteriorPredictiveReport(
        block_rmse=block_rmse,
        block_relative_rmse=block_relative_rmse,
        block_max_abs_residual=block_max_abs_residual,
        worst_relative_rmse=max(block_relative_rmse.values(), default=0.0),
    )


def run_v4_lite_profile_likelihood(
    objective: V4LiteObjective,
    vector: np.ndarray,
    *,
    profile_scales: np.ndarray,
    n_points: int,
    max_dimensions: int,
) -> tuple[V4LiteProfilePoint, ...]:
    cfg.require(n_points >= 3, "profile likelihood requires at least three points.")
    capped_dimensions = min(max_dimensions, int(vector.size))
    offsets = np.linspace(-1.0, 1.0, num=n_points, dtype=float)
    points: list[V4LiteProfilePoint] = []
    for dimension_index in range(capped_dimensions):
        for offset in offsets.tolist():
            trial = np.asarray(vector, dtype=float).copy()
            trial[dimension_index] += offset * profile_scales[dimension_index]
            value = objective.evaluate_vector(trial).total_objective
            points.append(
                V4LiteProfilePoint(
                    dimension_index=int(dimension_index),
                    offset=float(offset),
                    objective_value=float(value),
                )
            )
    return tuple(points)


def run_v4_lite_fake_data_recovery(
    objective: V4LiteObjective,
    truth_vector: np.ndarray,
    optimizer,
    *,
    n_restarts: int,
) -> V4LiteFakeDataRecoveryReport:
    truth = objective.evaluate_vector(truth_vector, return_artifacts=True)
    cfg.require(truth.artifacts is not None, "Truth evaluation must return artifacts for fake-data recovery.")
    synthetic_objective = objective.with_observed_summary(truth.artifacts.prediction.summary.align_to(objective.observed_summary))
    recovered_vector, recovered_value = optimizer(
        synthetic_objective,
        synthetic_objective.adapter.default_vector(),
        n_restarts,
    )
    scale = np.maximum(synthetic_objective.adapter.proposal_scales(), 1e-6)
    normalized_error = float(np.linalg.norm((recovered_vector - truth_vector) / scale) / np.sqrt(max(1, truth_vector.size)))
    return V4LiteFakeDataRecoveryReport(
        recovered_objective=float(recovered_value),
        normalized_error=normalized_error,
        passed=bool(normalized_error <= 1.5),
    )
