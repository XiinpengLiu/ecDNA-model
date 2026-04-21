"""
Observation-layer simulation for flow sorting, qPCDR, and ecTAG readouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import config as cfg

if TYPE_CHECKING:
    from cell import Cell


@dataclass(frozen=True)
class GateAssignment:
    cell: "Cell"
    latent_gate: int
    observed_gate: int


def _empty_copy_matrix() -> np.ndarray:
    return np.zeros((0, cfg.N_SPECIES), dtype=int)


def _copy_matrix_to_list(matrix: np.ndarray) -> list[list[int]]:
    if matrix.size == 0:
        return []
    return np.asarray(matrix, dtype=int).tolist()


def _gamma_poisson_sample(mean: float, overdispersion: float, rng: np.random.Generator) -> int:
    cfg.require(mean >= 0.0, "Observation mean must be non-negative.")
    cfg.require(overdispersion >= 0.0, "Observation overdispersion must be non-negative.")
    if mean == 0.0:
        return 0
    if overdispersion == 0.0:
        return int(rng.poisson(mean))
    shape = 1.0 / overdispersion
    scale = mean / shape
    latent_rate = rng.gamma(shape=shape, scale=scale)
    return int(rng.poisson(latent_rate))


def _histogram(values: np.ndarray, upper_bound: int) -> list[int]:
    clipped = np.clip(np.asarray(values, dtype=int), 0, upper_bound)
    counts = np.bincount(clipped, minlength=upper_bound + 1)
    return counts.astype(int).tolist()


def summarize_copy_statistics(copy_matrices_by_gate: dict[str, np.ndarray]) -> dict[str, dict]:
    copy_means_by_gate: dict[str, list[float]] = {}
    copy_vars_by_gate: dict[str, list[float]] = {}
    copy_histograms_by_gate: dict[str, dict[str, list[int]]] = {}
    zero_fraction_by_gate: dict[str, list[float]] = {}
    tail_fraction_by_gate: dict[str, list[float]] = {}
    joint_copy_correlations_by_gate: dict[str, list[list[float]]] = {}

    for gate_name in cfg.STATE_NAMES:
        matrix = np.asarray(copy_matrices_by_gate.get(gate_name, _empty_copy_matrix()), dtype=int)
        if matrix.size == 0:
            copy_means_by_gate[gate_name] = np.zeros(cfg.N_SPECIES, dtype=float).tolist()
            copy_vars_by_gate[gate_name] = np.zeros(cfg.N_SPECIES, dtype=float).tolist()
            copy_histograms_by_gate[gate_name] = {species: [0] for species in cfg.SPECIES}
            zero_fraction_by_gate[gate_name] = np.zeros(cfg.N_SPECIES, dtype=float).tolist()
            tail_fraction_by_gate[gate_name] = np.zeros(cfg.N_SPECIES, dtype=float).tolist()
            joint_copy_correlations_by_gate[gate_name] = np.zeros((cfg.N_SPECIES, cfg.N_SPECIES), dtype=float).tolist()
            continue

        copy_means_by_gate[gate_name] = np.mean(matrix, axis=0).astype(float).tolist()
        copy_vars_by_gate[gate_name] = np.var(matrix, axis=0).astype(float).tolist()
        zero_fraction_by_gate[gate_name] = np.mean(matrix == 0, axis=0).astype(float).tolist()

        tail_fraction = np.zeros(cfg.N_SPECIES, dtype=float)
        histograms: dict[str, list[int]] = {}
        for species_idx, species_name in enumerate(cfg.SPECIES):
            species_values = matrix[:, species_idx]
            threshold = float(np.quantile(species_values, 0.90, method="higher"))
            tail_fraction[species_idx] = float(np.mean(species_values >= threshold))
            histograms[species_name] = _histogram(species_values, int(np.max(species_values)))
        tail_fraction_by_gate[gate_name] = tail_fraction.tolist()
        copy_histograms_by_gate[gate_name] = histograms

        if matrix.shape[0] < 2 or np.any(np.std(matrix.astype(float), axis=0) == 0.0):
            correlation = np.zeros((cfg.N_SPECIES, cfg.N_SPECIES), dtype=float)
        else:
            correlation = np.corrcoef(matrix.astype(float), rowvar=False)
            correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        joint_copy_correlations_by_gate[gate_name] = correlation.astype(float).tolist()

    return {
        "copy_means_by_gate": copy_means_by_gate,
        "copy_vars_by_gate": copy_vars_by_gate,
        "copy_histograms_by_gate": copy_histograms_by_gate,
        "zero_fraction_by_gate": zero_fraction_by_gate,
        "tail_fraction_by_gate": tail_fraction_by_gate,
        "joint_copy_correlations_by_gate": joint_copy_correlations_by_gate,
    }


def _sample_gate_bias(params: cfg.ObservationParameters, rng: np.random.Generator) -> np.ndarray:
    if params.flow_overdispersion == 0.0:
        return np.ones(cfg.N_STATES, dtype=float)
    bias = np.exp(rng.normal(loc=0.0, scale=params.flow_overdispersion, size=cfg.N_STATES))
    cfg.require(np.all(np.isfinite(bias)) and np.all(bias > 0.0), "Gate bias must be finite and positive.")
    return bias.astype(float)


def sample_observed_gate(
    soft_state: np.ndarray,
    params: cfg.ObservationParameters,
    rng: np.random.Generator,
    gate_bias: np.ndarray | None = None,
) -> tuple[int, int]:
    latent_gate = int(rng.choice(np.arange(cfg.N_STATES), p=np.asarray(soft_state, dtype=float)))
    purity_column = np.asarray(params.sort_purity_matrix[:, latent_gate], dtype=float)
    if gate_bias is not None:
        purity_column = purity_column * gate_bias
    purity_total = float(np.sum(purity_column))
    cfg.require(purity_total > 0.0, "Observed gate probabilities must sum to a positive value.")
    observed_probs = purity_column / purity_total
    observed_gate = int(rng.choice(np.arange(cfg.N_STATES), p=observed_probs))
    return latent_gate, observed_gate


def simulate_flow_counts(
    cells: list["Cell"],
    obs_params: cfg.ObservationParameters,
    rng: np.random.Generator,
) -> list[GateAssignment]:
    gate_bias = _sample_gate_bias(obs_params, rng)
    assignments: list[GateAssignment] = []
    for cell in cells:
        latent_gate, observed_gate = sample_observed_gate(cell.soft_state, obs_params, rng, gate_bias=gate_bias)
        assignments.append(GateAssignment(cell=cell, latent_gate=latent_gate, observed_gate=observed_gate))
    return assignments


def simulate_sorted_qpcdr(
    cells_by_gate: dict[str, list["Cell"]],
    obs_params: cfg.ObservationParameters,
    rng: np.random.Generator,
) -> dict[str, dict]:
    raw_values: dict[str, dict[str, list[float]]] = {}
    mean_values: dict[str, list[float]] = {}
    for gate_name in cfg.STATE_NAMES:
        gate_cells = cells_by_gate.get(gate_name, [])
        gate_species: dict[str, list[float]] = {species: [] for species in cfg.SPECIES}
        gate_means = np.zeros(cfg.N_SPECIES, dtype=float)
        for species_idx, species_name in enumerate(cfg.SPECIES):
            if gate_cells:
                gate_copy_mean = float(np.mean([cell.copy_numbers[species_idx] for cell in gate_cells]))
                observed = [
                    float(
                        obs_params.qpcdr_intercept[species_idx]
                        + obs_params.qpcdr_slope[species_idx] * gate_copy_mean
                        + obs_params.qpcdr_sigma[species_idx] * rng.normal()
                    )
                ]
                gate_species[species_name] = observed
                gate_means[species_idx] = float(np.mean(observed))
        raw_values[gate_name] = gate_species
        mean_values[gate_name] = gate_means.tolist()
    return {"values": raw_values, "means": mean_values}


def simulate_sorted_ecTAG(
    cells_by_gate: dict[str, list["Cell"]],
    obs_params: cfg.ObservationParameters,
    rng: np.random.Generator,
) -> dict[str, dict]:
    raw_values: dict[str, dict[str, list[int]]] = {}
    histograms: dict[str, dict[str, list[int]]] = {}
    mean_values: dict[str, list[float]] = {}
    for gate_name in cfg.STATE_NAMES:
        gate_cells = cells_by_gate.get(gate_name, [])
        gate_species_values: dict[str, list[int]] = {}
        gate_histograms: dict[str, list[int]] = {}
        gate_means = np.zeros(cfg.N_SPECIES, dtype=float)
        for species_idx, species_name in enumerate(cfg.SPECIES):
            if gate_cells:
                counts = [
                    min(
                        _gamma_poisson_sample(
                            mean=float(
                                obs_params.ecTAG_detection_efficiency[species_idx] * cell.copy_numbers[species_idx]
                                + obs_params.ecTAG_background[species_idx]
                            ),
                            overdispersion=float(obs_params.ecTAG_overdispersion[species_idx]),
                            rng=rng,
                        ),
                        int(obs_params.ecTAG_max_observed),
                    )
                    for cell in gate_cells
                ]
                gate_species_values[species_name] = counts
                gate_histograms[species_name] = _histogram(np.asarray(counts, dtype=int), int(obs_params.ecTAG_max_observed))
                gate_means[species_idx] = float(np.mean(counts))
            else:
                gate_species_values[species_name] = []
                gate_histograms[species_name] = [0] * (int(obs_params.ecTAG_max_observed) + 1)
        raw_values[gate_name] = gate_species_values
        histograms[gate_name] = gate_histograms
        mean_values[gate_name] = gate_means.tolist()
    return {"values": raw_values, "histograms": histograms, "means": mean_values}


def make_observation_snapshot(
    cells: list["Cell"],
    truth_snapshot: dict,
    obs_params: cfg.ObservationParameters,
    rng: np.random.Generator,
) -> dict:
    assignments = simulate_flow_counts(cells, obs_params, rng)
    cells_by_gate: dict[str, list["Cell"]] = {state_name: [] for state_name in cfg.STATE_NAMES}
    latent_gate_counts = np.zeros(cfg.N_STATES, dtype=int)
    observed_gate_counts = np.zeros(cfg.N_STATES, dtype=int)

    for assignment in assignments:
        latent_gate_counts[assignment.latent_gate] += 1
        observed_gate_counts[assignment.observed_gate] += 1
        cells_by_gate[cfg.STATE_NAMES[assignment.observed_gate]].append(assignment.cell)

    copy_matrices_by_gate = {
        gate_name: (
            np.stack([cell.copy_numbers for cell in gate_cells], axis=0).astype(int)
            if gate_cells
            else _empty_copy_matrix()
        )
        for gate_name, gate_cells in cells_by_gate.items()
    }
    copy_summary = summarize_copy_statistics(copy_matrices_by_gate)
    qpcdr_summary = simulate_sorted_qpcdr(cells_by_gate, obs_params, rng)
    ectag_summary = simulate_sorted_ecTAG(cells_by_gate, obs_params, rng)

    observed_count = _gamma_poisson_sample(
        mean=float(len(cells)),
        overdispersion=float(obs_params.count_overdispersion),
        rng=rng,
    )
    pooled_qpcdr = np.zeros(cfg.N_SPECIES, dtype=float)
    pooled_ectag = np.zeros(cfg.N_SPECIES, dtype=float)
    if cells:
        for species_idx, species_name in enumerate(cfg.SPECIES):
            qpcdr_values = [
                value
                for gate_name in cfg.STATE_NAMES
                for value in qpcdr_summary["values"][gate_name][species_name]
            ]
            ectag_values = [
                value
                for gate_name in cfg.STATE_NAMES
                for value in ectag_summary["values"][gate_name][species_name]
            ]
            pooled_qpcdr[species_idx] = float(np.mean(qpcdr_values)) if qpcdr_values else 0.0
            pooled_ectag[species_idx] = float(np.mean(ectag_values)) if ectag_values else 0.0

    return {
        "observed_count": int(observed_count),
        "latent_gate_counts": latent_gate_counts.astype(int).tolist(),
        "latent_gate_fractions": (
            (latent_gate_counts / max(1, len(cells))).astype(float).tolist()
            if cells
            else np.zeros(cfg.N_STATES, dtype=float).tolist()
        ),
        "flow_counts": observed_gate_counts.astype(int).tolist(),
        "flow_fractions": (
            (observed_gate_counts / max(1, len(cells))).astype(float).tolist()
            if cells
            else np.zeros(cfg.N_STATES, dtype=float).tolist()
        ),
        "sorted_state_counts": {gate_name: int(len(gate_cells)) for gate_name, gate_cells in cells_by_gate.items()},
        "sorted_bulk_copy_means": copy_summary["copy_means_by_gate"],
        "sorted_copy_distributions": {
            gate_name: _copy_matrix_to_list(matrix)
            for gate_name, matrix in copy_matrices_by_gate.items()
        },
        "sorted_qpcdr": qpcdr_summary,
        "sorted_ecTAG": ectag_summary,
        "pooled_qpcdr_means": np.asarray(pooled_qpcdr, dtype=float).tolist(),
        "pooled_ecTAG_means": np.asarray(pooled_ectag, dtype=float).tolist(),
        "sort_purity_matrix": np.asarray(obs_params.sort_purity_matrix, dtype=float).tolist(),
        **copy_summary,
        "truth_soft_state_fractions": truth_snapshot["soft_state_fractions"],
    }
