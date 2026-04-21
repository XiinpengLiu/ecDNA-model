"""
Observed-summary extraction for fitting and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

from ecdna_model import config as cfg
from ecdna_model.core.simulation import SimulationResult
from ecdna_model.fit.data import CanonicalFitDataset, EcTAGRecord, QPCDRRecord
from ecdna_model.fit.simulation_runner import SimulationRunSet


TAIL_THRESHOLDS = (8, 16)


@dataclass(frozen=True)
class SummaryBlock:
    name: str
    keys: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        cfg.require(len(self.keys) == int(self.values.size), f"SummaryBlock {self.name} key/value length mismatch.")

    def as_mapping(self) -> dict[str, float]:
        return {key: float(value) for key, value in zip(self.keys, self.values.tolist())}

    def align_to(self, reference: "SummaryBlock") -> "SummaryBlock":
        mapping = self.as_mapping()
        missing = [key for key in reference.keys if key not in mapping]
        cfg.require(not missing, f"Summary block {self.name} is missing keys required by the observed data: {missing[:5]}.")
        return SummaryBlock(
            name=self.name,
            keys=reference.keys,
            values=np.array([mapping[key] for key in reference.keys], dtype=float),
        )


@dataclass(frozen=True)
class SummaryCollection:
    blocks: dict[str, SummaryBlock]

    def block_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.blocks))

    def align_to(self, reference: "SummaryCollection") -> "SummaryCollection":
        aligned: dict[str, SummaryBlock] = {}
        for block_name, reference_block in reference.blocks.items():
            cfg.require(block_name in self.blocks, f"Summary collection is missing block {block_name}.")
            aligned[block_name] = self.blocks[block_name].align_to(reference_block)
        return SummaryCollection(blocks=aligned)

    @classmethod
    def from_block_maps(cls, block_maps: dict[str, dict[str, float]]) -> "SummaryCollection":
        blocks: dict[str, SummaryBlock] = {}
        for block_name, mapping in block_maps.items():
            if not mapping:
                continue
            keys = tuple(sorted(mapping))
            values = np.array([mapping[key] for key in keys], dtype=float)
            blocks[block_name] = SummaryBlock(name=block_name, keys=keys, values=values)
        return cls(blocks=blocks)


def _week_from_time(time_value: float) -> int:
    return int(round(float(time_value))) + 1


def _histogram_probabilities(values: Iterable[int], upper_bound: int) -> np.ndarray:
    array = np.asarray(list(values), dtype=int)
    if array.size == 0:
        return np.zeros(upper_bound + 1, dtype=float)
    clipped = np.clip(array, 0, upper_bound)
    counts = np.bincount(clipped, minlength=upper_bound + 1).astype(float)
    return counts / float(np.sum(counts))


def _ectag_moments(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "zero_fraction": 0.0,
            "mean": 0.0,
            "variance": 0.0,
            "cv": 0.0,
            "tail_ge_8": 0.0,
            "tail_ge_16": 0.0,
        }
    mean_value = float(np.mean(values))
    variance = float(np.var(values))
    cv = 0.0 if mean_value <= 0.0 else float(np.sqrt(variance) / mean_value)
    return {
        "zero_fraction": float(np.mean(values == 0)),
        "mean": mean_value,
        "variance": variance,
        "cv": cv,
        "tail_ge_8": float(np.mean(values >= 8)),
        "tail_ge_16": float(np.mean(values >= 16)),
    }


def _corr_pairs_from_matrix(matrix: np.ndarray) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    if matrix.shape[0] < 2 or np.any(np.std(matrix, axis=0) == 0.0):
        correlations = np.zeros((cfg.N_SPECIES, cfg.N_SPECIES), dtype=float)
    else:
        correlations = np.corrcoef(matrix, rowvar=False)
        correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
    for first, second in combinations(range(cfg.N_SPECIES), 2):
        result[(cfg.SPECIES[first], cfg.SPECIES[second])] = float(correlations[first, second])
    return result


def summarize_dataset(
    dataset: CanonicalFitDataset,
    *,
    condition_names: Iterable[str] | None = None,
    dynamic_only: bool = True,
) -> SummaryCollection:
    selected_conditions = set(dataset.condition_names() if condition_names is None else tuple(condition_names))
    upper_bound = dataset.ectag_upper_bound()
    block_maps = _empty_block_maps()

    flow_groups: dict[tuple[str, int, str], list[tuple[int | None, float | None, int | None]]] = {}
    for record in dataset.flow:
        if record.condition not in selected_conditions:
            continue
        if dynamic_only and record.week == 1:
            continue
        flow_groups.setdefault((record.condition, record.week, record.state), []).append(
            (record.count, record.fraction, record.total_events)
        )
    mean_counts: dict[tuple[str, int, str], float] = {}
    count_totals: dict[tuple[str, int], float] = {}
    mean_fractions: dict[tuple[str, int, str], float] = {}
    for (condition_name, week, state_name), rows in flow_groups.items():
        counts = [float(count) for count, _fraction, _total in rows if count is not None]
        fractions = [float(fraction) for _count, fraction, _total in rows if fraction is not None]
        if counts:
            mean_count = float(np.mean(counts))
            mean_counts[(condition_name, week, state_name)] = mean_count
            count_totals[(condition_name, week)] = count_totals.get((condition_name, week), 0.0) + mean_count
        if fractions:
            mean_fractions[(condition_name, week, state_name)] = float(np.mean(fractions))

    for key in flow_groups:
        condition_name, week, state_name = key
        flow_fraction_key = f"{condition_name}|week{week}|state={state_name}"
        if key in mean_fractions:
            block_maps["flow_fraction"][flow_fraction_key] = mean_fractions[key]
        elif key in mean_counts:
            total = count_totals[(condition_name, week)]
            cfg.require(total > 0.0, f"Flow count totals for {condition_name} week {week} must be positive.")
            block_maps["flow_fraction"][flow_fraction_key] = mean_counts[key] / total
        if key in mean_counts:
            block_maps["flow_count"][flow_fraction_key] = mean_counts[key]

    count_groups: dict[tuple[str, int], list[float]] = {}
    for record in dataset.counts:
        if record.condition not in selected_conditions:
            continue
        if dynamic_only and record.week == 1:
            continue
        count_groups.setdefault((record.condition, record.week), []).append(float(record.value))
    for (condition_name, week), rows in count_groups.items():
        block_maps["count_total"][f"{condition_name}|week{week}"] = float(np.mean(rows))

    qpcdr_groups: dict[tuple[str, int, str, str], list[float]] = {}
    for record in dataset.qpcdr:
        if record.condition not in selected_conditions:
            continue
        if dynamic_only and record.week == 1:
            continue
        qpcdr_groups.setdefault((record.condition, record.week, record.state, record.species), []).append(float(record.value))
    for (condition_name, week, state_name, species_name), values in qpcdr_groups.items():
        prefix = f"{condition_name}|week{week}|state={state_name}|species={species_name}"
        array = np.asarray(values, dtype=float)
        block_maps["qpcdr"][f"{prefix}|mean"] = float(np.mean(array))
        if dataset.qpcdr_scale() == "copy_number":
            block_maps["qpcdr"][f"{prefix}|log_mean"] = float(np.mean(np.log1p(np.clip(array, 0.0, None))))

    ectag_groups: dict[tuple[str, int, str, str], list[int]] = {}
    ectag_cell_groups: dict[tuple[str, int, str, str], dict[str, int]] = {}
    for record in dataset.ectag:
        if record.condition not in selected_conditions:
            continue
        if dynamic_only and record.week == 1:
            continue
        ectag_groups.setdefault((record.condition, record.week, record.state, record.species), []).append(int(record.value))
        cell_key = f"{record.condition}|week{record.week}|state={record.state}|cell={record.cell_id}"
        ectag_cell_groups.setdefault((record.condition, record.week, record.state, cell_key), {})[record.species] = int(record.value)

    for (condition_name, week, state_name, species_name), values in ectag_groups.items():
        prefix = f"{condition_name}|week{week}|state={state_name}|species={species_name}"
        array = np.asarray(values, dtype=float)
        for metric_name, metric_value in _ectag_moments(array).items():
            block_maps["ectag_moments"][f"{prefix}|{metric_name}"] = metric_value
        histogram = _histogram_probabilities(values, upper_bound)
        for bin_index, probability in enumerate(histogram.tolist()):
            block_maps["ectag_hist"][f"{prefix}|bin={bin_index}"] = float(probability)

    corr_groups: dict[tuple[str, int, str], list[list[int]]] = {}
    for (condition_name, week, state_name, _cell_key), species_map in ectag_cell_groups.items():
        if set(species_map) == set(cfg.SPECIES):
            corr_groups.setdefault((condition_name, week, state_name), []).append(
                [species_map[species_name] for species_name in cfg.SPECIES]
            )
    for (condition_name, week, state_name), rows in corr_groups.items():
        correlations = _corr_pairs_from_matrix(np.asarray(rows, dtype=float))
        for (species_a, species_b), correlation in correlations.items():
            key = f"{condition_name}|week{week}|state={state_name}|pair={species_a}-{species_b}"
            block_maps["ectag_corr"][key] = correlation

    return SummaryCollection.from_block_maps(block_maps)


def summarize_simulation_result(
    condition_name: str,
    result: SimulationResult,
    dataset: CanonicalFitDataset,
    *,
    dynamic_only: bool = True,
    observed_layer: bool = True,
) -> SummaryCollection:
    upper_bound = dataset.ectag_upper_bound()
    block_maps = _empty_block_maps()
    snapshots = result.observations if observed_layer else result.truth_snapshots
    for time_value, snapshot in zip(result.times, snapshots):
        week = _week_from_time(time_value)
        if dynamic_only and week == 1:
            continue

        if observed_layer:
            for state_index, state_name in enumerate(cfg.STATE_NAMES):
                key_prefix = f"{condition_name}|week{week}|state={state_name}"
                block_maps["flow_fraction"][key_prefix] = float(snapshot["flow_fractions"][state_index])
                block_maps["flow_count"][key_prefix] = float(snapshot["flow_counts"][state_index])
            block_maps["count_total"][f"{condition_name}|week{week}"] = float(snapshot["observed_count"])
            for state_name in cfg.STATE_NAMES:
                for species_name in cfg.SPECIES:
                    prefix = f"{condition_name}|week{week}|state={state_name}|species={species_name}"
                    q_values = np.asarray(snapshot["sorted_qpcdr"]["values"][state_name][species_name], dtype=float)
                    block_maps["qpcdr"][f"{prefix}|mean"] = float(np.mean(q_values)) if q_values.size else 0.0
                    if dataset.qpcdr_scale() == "copy_number":
                        block_maps["qpcdr"][f"{prefix}|log_mean"] = (
                            float(np.mean(np.log1p(np.clip(q_values, 0.0, None)))) if q_values.size else 0.0
                        )
                    ectag_values = np.asarray(snapshot["sorted_ecTAG"]["values"][state_name][species_name], dtype=float)
                    for metric_name, metric_value in _ectag_moments(ectag_values).items():
                        block_maps["ectag_moments"][f"{prefix}|{metric_name}"] = metric_value
                    histogram = _histogram_probabilities(ectag_values.astype(int), upper_bound)
                    for bin_index, probability in enumerate(histogram.tolist()):
                        block_maps["ectag_hist"][f"{prefix}|bin={bin_index}"] = float(probability)

                aligned_lengths = [len(snapshot["sorted_ecTAG"]["values"][state_name][species_name]) for species_name in cfg.SPECIES]
                if len(set(aligned_lengths)) == 1:
                    if aligned_lengths[0] > 0:
                        matrix = np.stack(
                            [
                                np.asarray(snapshot["sorted_ecTAG"]["values"][state_name][species_name], dtype=float)
                                for species_name in cfg.SPECIES
                            ],
                            axis=1,
                        )
                    else:
                        matrix = np.zeros((0, cfg.N_SPECIES), dtype=float)
                    correlations = _corr_pairs_from_matrix(matrix)
                    for (species_a, species_b), correlation in correlations.items():
                        key = f"{condition_name}|week{week}|state={state_name}|pair={species_a}-{species_b}"
                        block_maps["ectag_corr"][key] = correlation
                else:
                    matrix = np.stack(
                        [np.zeros(0, dtype=float) for _ in cfg.SPECIES],
                        axis=1,
                    )
                    correlations = _corr_pairs_from_matrix(matrix)
                    for (species_a, species_b), correlation in correlations.items():
                        key = f"{condition_name}|week{week}|state={state_name}|pair={species_a}-{species_b}"
                        block_maps["ectag_corr"][key] = correlation
        else:
            block_maps["truth_count"][f"{condition_name}|week{week}"] = float(snapshot["population_size"])
            for state_index, state_name in enumerate(cfg.STATE_NAMES):
                block_maps["truth_state_fraction"][f"{condition_name}|week{week}|state={state_name}"] = float(
                    snapshot["soft_state_fractions"][state_index]
                )
            for species_index, species_name in enumerate(cfg.SPECIES):
                block_maps["truth_bulk_copy"][f"{condition_name}|week{week}|species={species_name}"] = float(
                    snapshot["bulk_copy_means"][species_index]
                )
            for diagnostic_name in ("mean_stress_score", "mean_survival_score", "mean_division_hazard", "mean_death_hazard"):
                block_maps["truth_diagnostics"][f"{condition_name}|week{week}|{diagnostic_name}"] = float(snapshot[diagnostic_name])

    return SummaryCollection.from_block_maps(block_maps)


def summarize_simulation_runset(
    run_set: SimulationRunSet,
    dataset: CanonicalFitDataset,
    *,
    dynamic_only: bool = True,
    reference: SummaryCollection | None = None,
    observed_layer: bool = True,
) -> tuple[SummaryCollection, ...]:
    n_replicates = run_set.n_replicates()
    summaries: list[SummaryCollection] = []
    for replicate_index in range(n_replicates):
        merged_maps = _empty_block_maps()
        for condition_name in run_set.condition_names():
            condition_summary = summarize_simulation_result(
                condition_name,
                run_set.runs[condition_name][replicate_index],
                dataset,
                dynamic_only=dynamic_only,
                observed_layer=observed_layer,
            )
            for block_name, block in condition_summary.blocks.items():
                merged_maps[block_name].update(block.as_mapping())
        collection = SummaryCollection.from_block_maps(merged_maps)
        if reference is not None:
            collection = collection.align_to(reference)
        summaries.append(collection)
    return tuple(summaries)


def mean_summary_collection(summaries: tuple[SummaryCollection, ...]) -> SummaryCollection:
    cfg.require(bool(summaries), "At least one summary collection is required.")
    reference = summaries[0]
    block_maps = _empty_block_maps()
    for block_name in reference.block_names():
        stacked = np.stack([summary.blocks[block_name].values for summary in summaries], axis=0)
        mean_values = np.mean(stacked, axis=0)
        block_maps[block_name] = {
            key: float(value)
            for key, value in zip(reference.blocks[block_name].keys, mean_values.tolist())
        }
    return SummaryCollection.from_block_maps(block_maps)


def _empty_block_maps() -> dict[str, dict[str, float]]:
    return {
        "flow_fraction": {},
        "flow_count": {},
        "count_total": {},
        "qpcdr": {},
        "ectag_moments": {},
        "ectag_hist": {},
        "ectag_corr": {},
        "truth_count": {},
        "truth_state_fraction": {},
        "truth_bulk_copy": {},
        "truth_diagnostics": {},
    }
