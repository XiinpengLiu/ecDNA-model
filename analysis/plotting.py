"""
Minimal plotting utilities for the ecDNA model.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
from core.simulation import SimulationResult


STATE_COLORS = ("#2563eb", "#16a34a", "#f59e0b", "#dc2626")
SPECIES_COLORS = ("#1d4ed8", "#be123c", "#047857")
DIAGNOSTIC_TIMEPOINT_COUNT = 7
DIAGNOSTIC_TRAJECTORY_CELL_COUNT = 256
DIAGNOSTIC_DIVISION_INHERITANCE_MAX_POINTS = 50000
DIAGNOSTIC_PHASE_SPACE_MAX_POINTS = 50000
T87_COPY_NUMBER_TIMEPOINT_COUNT = 7
T87_EXPERIMENT_START_DAY = 14.0
T87_EXPERIMENT_END_DAY = 56.0
T87_FILTERED_DDPCR_SOURCE = (
    Path(__file__).resolve().parents[1] / "data" / "2026-05-04-ddPCR-T87-drug-treatment-days-28-35-42-filtered.csv"
)
T87_DDPCR_TARGET_TO_SPECIES = {"ecMyc": "MYC", "ecCDK4": "CDK4", "ecPDGFRA": "PDGFRA"}
T87_COPY_TARGET_START_DAYS = {"R500": 21.0}
T87_TREATMENT_GROUPS = (
    ("Palbociclib (CDK4i)", ("ctrl", "P10", "P50", "P250")),
    ("Ripretinib (PDGFRAi)", ("ctrl", "R20", "R100", "R500")),
)
T87_CONDITION_COLORS = {
    "ctrl": "#ff0066",
    "P10": "#60c2f3",
    "P50": "#138a88",
    "P250": "#3f0788",
    "R20": "#60c2f3",
    "R100": "#138a88",
    "R500": "#3f0788",
}
EVENT_COLORS = {
    "division": "#2563eb",
    "death": "#111827",
    "cycle": "#7c3aed",
    "ecDNA gain": "#16a34a",
    "ecDNA loss": "#f97316",
    "amplification": "#dc2626",
    "other": "#6b7280",
}


def _save(fig: plt.Figure, save_path: str | Path | None) -> plt.Figure:
    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def _safe_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return token or "value"


def _condition_dose_label(condition: str) -> str:
    if condition == "ctrl":
        return "Ctrl"
    _drug, dose = cfg.T87_CONDITION_TREATMENTS[condition]
    return f"{dose:g} nM"


def plot_results(result: SimulationResult, title: str = "ecDNA simulation", save_path: str | Path | None = None) -> plt.Figure:
    record_indices = _terminal_aligned_record_indices(result)
    times = np.asarray([result.times[idx] for idx in record_indices], dtype=float)
    state_fractions = np.asarray(result.soft_state_fractions, dtype=float)[record_indices]
    bulk = np.asarray(result.bulk_copy_means, dtype=float)[record_indices]
    population_sizes = np.asarray(result.population_sizes, dtype=float)[record_indices]
    mean_stress_scores = np.asarray(result.mean_stress_scores, dtype=float)[record_indices]
    mean_survival_scores = np.asarray(result.mean_survival_scores, dtype=float)[record_indices]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(times, population_sizes, color="#1f2937", linewidth=2)
    axes[0, 0].set_title("Population size")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Cells")

    for idx, state_name in enumerate(cfg.STATE_NAMES):
        axes[0, 1].plot(times, state_fractions[:, idx], linewidth=2, label=state_name)
    axes[0, 1].set_title("Soft state fractions")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Fraction")
    axes[0, 1].legend(frameon=False, fontsize=8)

    for idx, species in enumerate(cfg.SPECIES):
        axes[1, 0].plot(times, bulk[:, idx], linewidth=2, label=species)
    axes[1, 0].set_title("Bulk mean ecDNA copies")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Mean copies")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].plot(times, mean_stress_scores, label="Stress score", linewidth=2)
    axes[1, 1].plot(times, mean_survival_scores, label="Survival score", linewidth=2)
    axes[1, 1].set_title("Latent stress and survival")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].legend(frameon=False, fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_observation_proxies(result: SimulationResult, save_path: str | Path | None = None) -> plt.Figure:
    record_indices = _terminal_aligned_record_indices(result)
    times = np.asarray([result.times[idx] for idx in record_indices], dtype=float)
    selected_observations = [result.observations[idx] for idx in record_indices]
    flow_fractions = np.asarray([snapshot["flow_fractions"] for snapshot in selected_observations], dtype=float)
    qpcdr_means = np.asarray([snapshot["pooled_qpcdr_means"] for snapshot in selected_observations], dtype=float)
    counts = np.asarray([snapshot["observed_count"] for snapshot in selected_observations], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].stackplot(times, flow_fractions.T, labels=cfg.STATE_NAMES)
    axes[0].set_title("Observed flow fractions")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Fraction")

    for idx, species in enumerate(cfg.SPECIES):
        axes[1].plot(times, qpcdr_means[:, idx], linewidth=2, label=species)
    axes[1].set_title("Sorted qPCDR proxy")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Observed signal")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(times, counts, color="#7c3aed", linewidth=2)
    axes[2].set_title("Observed cell count")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Count")

    fig.tight_layout()
    return _save(fig, save_path)


def plot_event_summary(result: SimulationResult, save_path: str | Path | None = None) -> plt.Figure:
    event_counts: dict[str, int] = {}
    for _, event_type, _, _ in result.events:
        event_counts[event_type] = event_counts.get(event_type, 0) + 1

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(event_counts.keys()) if event_counts else ["none"]
    values = [event_counts[key] for key in labels] if event_counts else [0]
    ax.bar(labels, values, color="#2563eb")
    ax.set_title("Event counts")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return _save(fig, save_path)


def _dominant_state_index(soft_state: list[float] | np.ndarray) -> int:
    composition = np.asarray(soft_state, dtype=float)
    cfg.require(composition.shape == (cfg.N_STATES,), "Expected a full 4-state soft-state vector.")
    return int(np.argmax(composition))


def _dominant_state_from_cell(cell: dict) -> int:
    if "dominant_state_index" in cell:
        return int(cell["dominant_state_index"])
    if "dominant_state" in cell:
        return int(cfg.STATE_INDEX[str(cell["dominant_state"])])
    return _dominant_state_index(cell["soft_state"])


def _state_count_matrix(result: SimulationResult) -> np.ndarray:
    counts = np.zeros((len(result.times), cfg.N_STATES), dtype=float)
    for idx, snapshot in enumerate(result.truth_snapshots):
        if "dominant_state_counts" in snapshot:
            counts[idx, :] = np.asarray(snapshot["dominant_state_counts"], dtype=float)
        else:
            counts[idx, :] = result.population_sizes[idx] * np.asarray(result.soft_state_fractions[idx], dtype=float)
    return counts


def _state_copy_mean_tensor(result: SimulationResult) -> np.ndarray:
    values = np.zeros((len(result.times), cfg.N_STATES, cfg.N_SPECIES), dtype=float)
    for time_idx, snapshot in enumerate(result.truth_snapshots):
        by_gate = snapshot.get("copy_means_by_gate", {})
        for state_idx, state_name in enumerate(cfg.STATE_NAMES):
            values[time_idx, state_idx, :] = np.asarray(
                by_gate.get(state_name, np.zeros(cfg.N_SPECIES, dtype=float)),
                dtype=float,
            )
    return values


def _nonempty_snapshot_indices(result: SimulationResult) -> list[int]:
    return [idx for idx, snapshot in enumerate(result.cell_snapshots) if snapshot]


def _representative_snapshot_indices(result: SimulationResult, target_count: int | None = None) -> list[int]:
    nonempty = _nonempty_snapshot_indices(result)
    if not nonempty:
        return []
    if target_count is None or len(nonempty) <= target_count:
        return nonempty
    raw_positions = np.rint(np.linspace(0, len(nonempty) - 1, num=target_count)).astype(int)
    return [nonempty[pos] for pos in sorted(set(raw_positions.tolist()))]


def _terminal_aligned_indices_by_time(
    result: SimulationResult,
    candidate_indices: list[int],
    target_count: int | None = DIAGNOSTIC_TIMEPOINT_COUNT,
) -> list[int]:
    """Return all indices by default; optionally select a terminal-aligned subset."""
    if target_count is None:
        return list(candidate_indices)
    if target_count <= 0:
        return []
    if len(candidate_indices) <= target_count:
        return candidate_indices

    times = np.asarray([float(result.times[idx]) for idx in candidate_indices], dtype=float)
    target_times = np.linspace(float(times[0]), float(times[-1]), num=target_count)
    selected_positions: list[int] = []
    previous_position = -1

    for target_idx, target_time in enumerate(target_times):
        remaining_slots = target_count - target_idx - 1
        min_position = previous_position + 1
        max_position = len(candidate_indices) - remaining_slots - 1
        if target_idx == target_count - 1:
            position = len(candidate_indices) - 1
        else:
            candidate_times = times[min_position : max_position + 1]
            position = min_position + int(np.argmin(np.abs(candidate_times - target_time)))
        selected_positions.append(position)
        previous_position = position

    return [candidate_indices[pos] for pos in selected_positions]


def _terminal_aligned_record_indices(
    result: SimulationResult,
    target_count: int | None = DIAGNOSTIC_TIMEPOINT_COUNT,
) -> list[int]:
    return _terminal_aligned_indices_by_time(result, list(range(len(result.times))), target_count)


def _terminal_aligned_snapshot_indices(
    result: SimulationResult,
    target_count: int | None = DIAGNOSTIC_TIMEPOINT_COUNT,
) -> list[int]:
    return _terminal_aligned_indices_by_time(result, _nonempty_snapshot_indices(result), target_count)


def _blank_axis(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_population_state_counts(
    result: SimulationResult,
    title: str = "Population and state counts",
    save_path: str | Path | None = None,
) -> plt.Figure:
    record_indices = _terminal_aligned_record_indices(result)
    times = np.asarray([result.times[idx] for idx in record_indices], dtype=float)
    state_counts = _state_count_matrix(result)[record_indices]
    population_sizes = np.asarray(result.population_sizes, dtype=float)[record_indices]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times, population_sizes, color="#111827", linewidth=2.5, label="Total")
    for state_idx, state_name in enumerate(cfg.STATE_NAMES):
        ax.plot(times, state_counts[:, state_idx], color=STATE_COLORS[state_idx], linewidth=2, label=state_name)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cells")
    ax.legend(frameon=False, ncols=2, fontsize=8)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_state_fraction_trajectories(
    result: SimulationResult,
    title: str = "State fraction trajectories",
    save_path: str | Path | None = None,
) -> plt.Figure:
    record_indices = _terminal_aligned_record_indices(result)
    times = np.asarray([result.times[idx] for idx in record_indices], dtype=float)
    state_fractions = np.asarray(result.soft_state_fractions, dtype=float)[record_indices]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(times, state_fractions.T, labels=cfg.STATE_NAMES, colors=STATE_COLORS, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return _save(fig, save_path)


def plot_ecdna_mean_trajectories(
    result: SimulationResult,
    title: str = "ecDNA copy-number trajectories",
    save_path: str | Path | None = None,
) -> plt.Figure:
    record_indices = _terminal_aligned_record_indices(result)
    times = np.asarray([result.times[idx] for idx in record_indices], dtype=float)
    bulk = np.asarray(result.bulk_copy_means, dtype=float)[record_indices]
    by_state = _state_copy_mean_tensor(result)[record_indices]

    fig, axes = plt.subplots(1, cfg.N_SPECIES, figsize=(14, 4), sharex=True)
    for species_idx, species_name in enumerate(cfg.SPECIES):
        ax = axes[species_idx]
        ax.plot(times, bulk[:, species_idx], color="#111827", linewidth=2.5, linestyle="--", label="Bulk")
        for state_idx, state_name in enumerate(cfg.STATE_NAMES):
            ax.plot(
                times,
                by_state[:, state_idx, species_idx],
                color=STATE_COLORS[state_idx],
                linewidth=2,
                label=state_name,
            )
        ax.set_title(species_name)
        ax.set_xlabel("Time")
        ax.set_ylabel("Mean copies")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_ecdna_copy_distributions(
    result: SimulationResult,
    title: str = "ecDNA copy-number distributions",
    save_path: str | Path | None = None,
) -> plt.Figure:
    snapshot_indices = _terminal_aligned_snapshot_indices(result)

    fig, axes = plt.subplots(
        cfg.N_SPECIES,
        1,
        figsize=(10, 3.2 * cfg.N_SPECIES),
        squeeze=False,
    )

    if not snapshot_indices:
        for ax in axes.ravel():
            _blank_axis(ax, "No cell snapshots recorded")
        fig.suptitle(title)
        fig.tight_layout()
        return _save(fig, save_path)

    for species_idx, species_name in enumerate(cfg.SPECIES):
        ax = axes[species_idx, 0]
        time_labels: list[str] = []
        values_by_time: list[np.ndarray] = []
        for snapshot_idx in snapshot_indices:
            snapshot = result.cell_snapshots[snapshot_idx]
            copies = np.asarray([cell["copy_numbers"] for cell in snapshot], dtype=int)
            if copies.size == 0:
                continue
            values_by_time.append(copies[:, species_idx])
            time_labels.append(f"t={float(result.times[snapshot_idx]):.1f}")

        if not values_by_time:
            _blank_axis(ax, "No cell snapshots recorded")
            continue

        positions = np.arange(len(values_by_time), dtype=float)
        parts = ax.violinplot(values_by_time, positions=positions, showmeans=True, showextrema=False, widths=0.8)
        for body in parts["bodies"]:
            body.set_facecolor(SPECIES_COLORS[species_idx])
            body.set_edgecolor(SPECIES_COLORS[species_idx])
            body.set_alpha(0.35)
        parts["cmeans"].set_color("#111827")
        parts["cmeans"].set_linewidth(1.0)

        max_values = [float(np.max(values)) for values in values_by_time]
        ax.scatter(positions, max_values, s=18, color="#111827", zorder=3)
        ax.set_title(species_name)
        ax.set_xticks(positions, labels=time_labels, rotation=45, ha="right")
        ax.set_xlabel("Time")
        ax.set_ylabel("ecDNA copy number")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)

    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_path)


def _event_category(event_type: str) -> str:
    if event_type == "division":
        return "division"
    if event_type == "death":
        return "death"
    if event_type.startswith("gain_"):
        return "ecDNA gain"
    if event_type.startswith("loss_"):
        return "ecDNA loss"
    if "_to_" in event_type:
        return "cycle"
    return "other"


def _amplified_copy_count(details: dict) -> int:
    if not {"state_pre", "daughter_one", "daughter_two"}.issubset(details):
        return 0
    mother = np.asarray(details["state_pre"]["copy_numbers"], dtype=int)
    daughter_one = np.asarray(details["daughter_one"]["copy_numbers"], dtype=int)
    daughter_two = np.asarray(details["daughter_two"]["copy_numbers"], dtype=int)
    amplification = daughter_one + daughter_two - 2 * mother
    return int(np.sum(np.maximum(amplification, 0)))


def plot_event_counts_by_window(
    result: SimulationResult,
    title: str = "Event counts by time window",
    save_path: str | Path | None = None,
) -> plt.Figure:
    record_indices = _terminal_aligned_record_indices(result)
    times = np.asarray([result.times[idx] for idx in record_indices], dtype=float)
    if times.size == 0:
        fig, ax = plt.subplots(figsize=(9, 4))
        _blank_axis(ax, "No recorded times")
        return _save(fig, save_path)

    if times[0] <= 0.0:
        edges = times
    else:
        edges = np.concatenate(([0.0], times))
    if edges.size < 2:
        edges = np.asarray([0.0, float(times[-1])], dtype=float)
    if edges[0] == edges[-1]:
        edges[-1] = edges[0] + 1.0

    categories = tuple(EVENT_COLORS.keys())
    counts = {category: np.zeros(edges.size - 1, dtype=float) for category in categories}
    for event_time, event_type, _cell_id, details in result.events:
        window_idx = int(np.searchsorted(edges, float(event_time), side="left") - 1)
        if window_idx < 0 or window_idx >= edges.size - 1:
            continue
        counts[_event_category(event_type)][window_idx] += 1.0
        if event_type == "division":
            counts["amplification"][window_idx] += _amplified_copy_count(details)

    mids = 0.5 * (edges[:-1] + edges[1:])
    widths = np.maximum(edges[1:] - edges[:-1], 1e-6)
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(edges.size - 1, dtype=float)
    for category in categories:
        values = counts[category]
        if not np.any(values):
            continue
        ax.bar(mids, values, width=0.8 * widths, bottom=bottom, color=EVENT_COLORS[category], label=category)
        bottom += values
    ax.set_title(title)
    ax.set_xlabel("Time window")
    ax.set_ylabel("Events")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, save_path)


def _state_transition_counts(result: SimulationResult, snapshot_indices: list[int] | None = None) -> np.ndarray:
    transition_counts = np.zeros((cfg.N_STATES, cfg.N_STATES), dtype=float)
    indices = list(range(len(result.cell_snapshots))) if snapshot_indices is None else snapshot_indices

    previous_states: dict[int, int] | None = None
    for snapshot_idx in indices:
        snapshot = result.cell_snapshots[snapshot_idx]
        current_states = {int(cell["cell_id"]): _dominant_state_from_cell(cell) for cell in snapshot}
        if previous_states is not None:
            for cell_id, previous_state in previous_states.items():
                current_state = current_states.get(cell_id)
                if current_state is not None:
                    transition_counts[previous_state, current_state] += 1.0
        if current_states:
            previous_states = current_states

    for _event_time, event_type, _cell_id, details in result.events:
        if event_type != "division" or "state_pre" not in details:
            continue
        mother_state = _dominant_state_index(details["state_pre"]["soft_state"])
        for daughter_key in ("daughter_one", "daughter_two"):
            daughter = details.get(daughter_key)
            if daughter is None:
                continue
            daughter_state = _dominant_state_index(daughter["soft_state"])
            transition_counts[mother_state, daughter_state] += 1.0
    return transition_counts


def plot_state_transition_heatmap(
    result: SimulationResult,
    title: str = "Dominant-state transition counts",
    save_path: str | Path | None = None,
) -> plt.Figure:
    snapshot_indices = _terminal_aligned_snapshot_indices(result)
    transition_counts = _state_transition_counts(result, snapshot_indices)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(transition_counts, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_xticks(np.arange(cfg.N_STATES), labels=cfg.STATE_NAMES, rotation=30, ha="right")
    ax.set_yticks(np.arange(cfg.N_STATES), labels=cfg.STATE_NAMES)
    for source_idx in range(cfg.N_STATES):
        for target_idx in range(cfg.N_STATES):
            ax.text(
                target_idx,
                source_idx,
                f"{int(transition_counts[source_idx, target_idx])}",
                ha="center",
                va="center",
                color="#111827",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    return _save(fig, save_path)


def _cell_ids_from_latest_snapshot(result: SimulationResult, n_cells: int | None = None) -> list[int]:
    _latest_time, latest_snapshot = _latest_nonempty_snapshot(result)
    ids = sorted(int(cell["cell_id"]) for cell in latest_snapshot)
    if n_cells is None or len(ids) <= n_cells:
        return ids
    indices = np.rint(np.linspace(0, len(ids) - 1, num=n_cells)).astype(int)
    return [ids[idx] for idx in sorted(set(indices.tolist()))]


def plot_single_cell_trajectories(
    result: SimulationResult,
    n_cells: int | None = None,
    title: str = "Single-cell trajectories",
    save_path: str | Path | None = None,
) -> plt.Figure:
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0])
    state_ax = fig.add_subplot(gs[0, :])
    copy_axes = [fig.add_subplot(gs[1, idx]) for idx in range(cfg.N_SPECIES)]

    snapshot_indices = _terminal_aligned_snapshot_indices(result)
    if not snapshot_indices:
        _blank_axis(state_ax, "No cell snapshots recorded")
        for ax in copy_axes:
            _blank_axis(ax, "No cell snapshots recorded")
        fig.suptitle(title)
        fig.tight_layout()
        return _save(fig, save_path)

    selected_ids = _cell_ids_from_latest_snapshot(result, n_cells)
    times = np.asarray([result.times[idx] for idx in snapshot_indices], dtype=float)
    states = np.full((len(selected_ids), len(times)), np.nan, dtype=float)
    copy_values = np.full((len(selected_ids), len(times), cfg.N_SPECIES), np.nan, dtype=float)
    row_by_cell = {cell_id: row_idx for row_idx, cell_id in enumerate(selected_ids)}

    for time_idx, snapshot_idx in enumerate(snapshot_indices):
        snapshot = result.cell_snapshots[snapshot_idx]
        for cell in snapshot:
            row_idx = row_by_cell.get(int(cell["cell_id"]))
            if row_idx is None:
                continue
            states[row_idx, time_idx] = _dominant_state_from_cell(cell)
            copy_values[row_idx, time_idx, :] = np.asarray(cell["copy_numbers"], dtype=float)

    cmap = ListedColormap(STATE_COLORS)
    cmap.set_bad("#f3f4f6")
    state_ax.imshow(np.ma.masked_invalid(states), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=cfg.N_STATES - 1)
    state_ax.set_title("Dominant state")
    state_ax.set_xlabel("Time")
    state_ax.set_ylabel("Cell")
    state_ax.set_xticks(np.arange(len(times)), labels=[f"{time:.1f}" for time in times], rotation=45, ha="right")
    state_ax.set_yticks(np.arange(len(selected_ids)), labels=[str(cell_id) for cell_id in selected_ids])

    legend_handles = [
        plt.Line2D([0], [0], marker="s", linestyle="none", color=STATE_COLORS[idx], label=state_name)
        for idx, state_name in enumerate(cfg.STATE_NAMES)
    ]
    state_ax.legend(handles=legend_handles, frameon=False, ncols=4, fontsize=8, loc="upper right")

    for species_idx, species_name in enumerate(cfg.SPECIES):
        ax = copy_axes[species_idx]
        for row_idx in range(len(selected_ids)):
            ax.plot(times, copy_values[row_idx, :, species_idx], color=SPECIES_COLORS[species_idx], alpha=0.35, linewidth=1)
        ax.set_title(f"{species_name} copies")
        ax.set_xlabel("Time")
        ax.set_ylabel("Copy number")

    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_division_inheritance(
    result: SimulationResult,
    title: str = "Mother-daughter ecDNA inheritance",
    save_path: str | Path | None = None,
    max_points: int | None = None,
) -> plt.Figure:
    division_details: list[dict] = []
    for _event_time, event_type, _cell_id, details in result.events:
        if event_type != "division" or not {"state_pre", "daughter_one", "daughter_two"}.issubset(details):
            continue
        division_details.append(details)

    if max_points is not None and max_points > 0:
        max_divisions = max(1, int(np.ceil(max_points / 2.0)))
        if len(division_details) > max_divisions:
            indices = np.rint(np.linspace(0, len(division_details) - 1, num=max_divisions)).astype(int)
            division_details = [division_details[idx] for idx in sorted(set(indices.tolist()))]

    mother_values: list[list[int]] = []
    daughter_values: list[list[int]] = []
    for details in division_details:
        mother = np.asarray(details["state_pre"]["copy_numbers"], dtype=int)
        mother_values.append(mother.tolist())
        mother_values.append(mother.tolist())
        daughter_values.append(np.asarray(details["daughter_one"]["copy_numbers"], dtype=int).tolist())
        daughter_values.append(np.asarray(details["daughter_two"]["copy_numbers"], dtype=int).tolist())

    fig, axes = plt.subplots(1, cfg.N_SPECIES, figsize=(14, 4))
    if not mother_values:
        for ax in axes:
            _blank_axis(ax, "No division events recorded")
        fig.suptitle(title)
        fig.tight_layout()
        return _save(fig, save_path)

    mothers = np.asarray(mother_values, dtype=float)
    daughters = np.asarray(daughter_values, dtype=float)
    for species_idx, species_name in enumerate(cfg.SPECIES):
        ax = axes[species_idx]
        ax.scatter(
            mothers[:, species_idx],
            daughters[:, species_idx],
            s=18,
            alpha=0.45,
            color=SPECIES_COLORS[species_idx],
            edgecolors="none",
        )
        axis_max = float(max(np.max(mothers[:, species_idx]), np.max(daughters[:, species_idx]), 1.0))
        ax.plot([0, axis_max], [0, axis_max], color="#111827", linewidth=1, linestyle="--")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_title(species_name)
        ax.set_xlabel("Mother copies")
        ax.set_ylabel("Daughter copies")
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_state_ecdna_heatmaps(
    result: SimulationResult,
    title: str = "State-ecDNA mean-copy heatmaps",
    save_path: str | Path | None = None,
) -> plt.Figure:
    record_indices = _terminal_aligned_record_indices(result)
    times = np.asarray([result.times[idx] for idx in record_indices], dtype=float)
    by_state = _state_copy_mean_tensor(result)[record_indices]

    fig, axes = plt.subplots(1, cfg.N_SPECIES, figsize=(14, 4), sharey=True)
    for species_idx, species_name in enumerate(cfg.SPECIES):
        ax = axes[species_idx]
        image = ax.imshow(by_state[:, :, species_idx].T, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(species_name)
        ax.set_xlabel("Time")
        ax.set_xticks(np.arange(len(times)), labels=[f"{time:.1f}" for time in times], rotation=45, ha="right")
        ax.set_yticks(np.arange(cfg.N_STATES), labels=cfg.STATE_NAMES)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("State")
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_latent_phase_space(
    result: SimulationResult,
    title: str = "Latent burden-stress-survival phase space",
    save_path: str | Path | None = None,
    max_points: int | None = None,
) -> plt.Figure:
    snapshot_indices = _terminal_aligned_snapshot_indices(result)
    rows: list[dict] = [cell for snapshot_idx in snapshot_indices for cell in result.cell_snapshots[snapshot_idx]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if not rows:
        for ax in axes:
            _blank_axis(ax, "No cell snapshots recorded")
        fig.suptitle(title)
        fig.tight_layout()
        return _save(fig, save_path)

    if max_points is not None and len(rows) > max_points:
        indices = np.rint(np.linspace(0, len(rows) - 1, num=max_points)).astype(int)
        rows = [rows[idx] for idx in sorted(set(indices.tolist()))]

    copies = np.asarray([cell["copy_numbers"] for cell in rows], dtype=float)
    burdens = np.log1p(copies) @ cfg.DEFAULT_MODEL_PARAMETERS.exposure.burden_weights
    stress = np.asarray([cell["stress_score"] for cell in rows], dtype=float)
    survival = np.asarray([cell["survival_score"] for cell in rows], dtype=float)
    division_hazard = np.asarray([cell.get("division_hazard", 0.0) for cell in rows], dtype=float)
    death_hazard = np.asarray([cell.get("death_hazard", 0.0) for cell in rows], dtype=float)
    states = np.asarray([_dominant_state_from_cell(cell) for cell in rows], dtype=int)

    def scaled_sizes(values: np.ndarray) -> np.ndarray:
        if values.size == 0 or float(np.max(values)) <= 0.0:
            return np.full(values.shape, 18.0, dtype=float)
        return 18.0 + 70.0 * values / float(np.max(values))

    for state_idx, state_name in enumerate(cfg.STATE_NAMES):
        mask = states == state_idx
        axes[0].scatter(
            burdens[mask],
            stress[mask],
            s=scaled_sizes(division_hazard[mask]),
            color=STATE_COLORS[state_idx],
            alpha=0.35,
            edgecolors="none",
            label=state_name,
        )
        axes[1].scatter(
            stress[mask],
            survival[mask],
            s=scaled_sizes(death_hazard[mask]),
            color=STATE_COLORS[state_idx],
            alpha=0.35,
            edgecolors="none",
        )

    axes[0].set_title("Burden vs stress, size=division hazard")
    axes[0].set_xlabel("ecDNA burden")
    axes[0].set_ylabel("Stress score")
    axes[1].set_title("Stress vs survival, size=death hazard")
    axes[1].set_xlabel("Stress score")
    axes[1].set_ylabel("Survival score")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_single_run_diagnostic_suite(result: SimulationResult, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plot_specs = (
        ("01_population_state_counts.png", plot_population_state_counts),
        ("02_state_fractions.png", plot_state_fraction_trajectories),
        ("03_ecdna_mean_trajectories.png", plot_ecdna_mean_trajectories),
        ("04_ecdna_copy_distributions.png", plot_ecdna_copy_distributions),
        ("05_event_counts_by_window.png", plot_event_counts_by_window),
        ("06_state_transition_heatmap.png", plot_state_transition_heatmap),
        (
            "07_single_cell_trajectories.png",
            lambda run_result, save_path: plot_single_cell_trajectories(
                run_result,
                n_cells=DIAGNOSTIC_TRAJECTORY_CELL_COUNT,
                save_path=save_path,
            ),
        ),
        (
            "08_division_inheritance.png",
            lambda run_result, save_path: plot_division_inheritance(
                run_result,
                max_points=DIAGNOSTIC_DIVISION_INHERITANCE_MAX_POINTS,
                save_path=save_path,
            ),
        ),
        ("09_state_ecdna_heatmaps.png", plot_state_ecdna_heatmaps),
        (
            "10_latent_phase_space.png",
            lambda run_result, save_path: plot_latent_phase_space(
                run_result,
                max_points=DIAGNOSTIC_PHASE_SPACE_MAX_POINTS,
                save_path=save_path,
            ),
        ),
    )
    written: dict[str, Path] = {}
    for file_name, plot_func in plot_specs:
        path = output / file_name
        plot_func(result, save_path=path)
        written[file_name] = path
    return written


def _read_t87_metadata(condition_dir: Path) -> dict:
    for metadata_path in (condition_dir / "tables" / "metadata.json", condition_dir / "run_metadata.json"):
        if metadata_path.exists():
            return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def _load_t87_condition_frames(output_dir: Path, conditions: tuple[str, ...]) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    frames: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict] = {}
    for condition in conditions:
        condition_dir = output_dir / condition
        table_path = condition_dir / "tables" / "time_summary.csv"
        if not table_path.exists():
            continue
        frames[condition] = pd.read_csv(table_path).sort_values("time").reset_index(drop=True)
        metadata[condition] = _read_t87_metadata(condition_dir)
    return frames, metadata


def _load_t87_week1_cell_counts(raw_dir: str | Path | None) -> dict[str, float]:
    if raw_dir is None:
        return {}
    path = Path(raw_dir) / "cell_count.csv"
    if not path.exists():
        return {}
    cell_count = pd.read_csv(path)
    if not {"week", "condition", "total_cell_count"}.issubset(cell_count.columns):
        return {}
    week1 = cell_count[cell_count["week"].astype(int) == 1].copy()
    if week1.empty:
        return {}
    return week1.groupby("condition")["total_cell_count"].median().astype(float).to_dict()


def _metadata_n_init(metadata: dict) -> float:
    simulation = metadata.get("simulation", {}) if isinstance(metadata, dict) else {}
    if not isinstance(simulation, dict):
        return float("nan")
    try:
        return float(simulation.get("n_init", float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def _condition_cell_count_scale(
    condition: str,
    frame: pd.DataFrame,
    metadata: dict,
    week1_cell_counts: dict[str, float],
) -> float:
    raw_count = float(week1_cell_counts.get(condition, float("nan")))
    if not np.isfinite(raw_count) or raw_count <= 0.0:
        return 1.0

    denominator = _metadata_n_init(metadata)
    if not np.isfinite(denominator) or denominator <= 0.0:
        if frame.empty or "population_size" not in frame:
            return 1.0
        denominator = float(frame["population_size"].iloc[0])
    if not np.isfinite(denominator) or denominator <= 0.0:
        return 1.0
    return raw_count / denominator


def _t87_copy_target_start_day(condition: str) -> float:
    return float(T87_COPY_TARGET_START_DAYS.get(condition, T87_EXPERIMENT_START_DAY))


def _t87_sim_time_to_aligned_day(time: float) -> float:
    span = T87_EXPERIMENT_END_DAY - T87_EXPERIMENT_START_DAY
    return T87_EXPERIMENT_START_DAY + span * float(time) / float(cfg.T87_TREATMENT_END_TIME)


def _t87_experimental_day_to_aligned_day(condition: str, day: float) -> float:
    start_day = _t87_copy_target_start_day(condition)
    source_span = T87_EXPERIMENT_END_DAY - start_day
    if source_span <= 0.0:
        return T87_EXPERIMENT_START_DAY
    target_span = T87_EXPERIMENT_END_DAY - T87_EXPERIMENT_START_DAY
    return T87_EXPERIMENT_START_DAY + target_span * (float(day) - start_day) / source_span


def _empty_t87_copy_targets() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "condition",
            "day",
            "aligned_day",
            "species",
            "ddpcr_copy_number",
            "ddpcr_sd_or_ci",
        ]
    )


def _load_t87_filtered_copy_targets(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty_t87_copy_targets()

    raw = pd.read_csv(path)
    required = {"Sample", "Target", "CNV"}
    if not required.issubset(raw.columns):
        return _empty_t87_copy_targets()

    parsed = raw["Sample"].astype(str).str.extract(r"^d(?P<day>\d+)\s+(?P<condition>\S+)$")
    rows = raw.join(parsed)
    rows = rows[rows["condition"].isin(cfg.T87_CONDITION_TREATMENTS) & rows["Target"].isin(T87_DDPCR_TARGET_TO_SPECIES)].copy()
    if rows.empty:
        return _empty_t87_copy_targets()

    rows["day"] = rows["day"].astype(float)
    rows["species"] = rows["Target"].map(T87_DDPCR_TARGET_TO_SPECIES)
    rows["ddpcr_copy_number"] = rows["CNV"].astype(float)
    if {"PoissonCNVMin", "PoissonCNVMax"}.issubset(rows.columns):
        rows["ddpcr_sd_or_ci"] = (
            rows["PoissonCNVMax"].astype(float) - rows["PoissonCNVMin"].astype(float)
        ) / 2.0
    else:
        rows["ddpcr_sd_or_ci"] = float("nan")

    start_days = rows["condition"].map(_t87_copy_target_start_day).astype(float)
    rows = rows[(rows["day"] >= start_days) & (rows["day"] <= T87_EXPERIMENT_END_DAY)].copy()
    if rows.empty:
        return _empty_t87_copy_targets()

    grouped = (
        rows.groupby(["condition", "day", "species"], as_index=False)
        .agg(ddpcr_copy_number=("ddpcr_copy_number", "median"), ddpcr_sd_or_ci=("ddpcr_sd_or_ci", "median"))
        .sort_values(["condition", "species", "day"])
    )
    grouped["aligned_day"] = [
        _t87_experimental_day_to_aligned_day(str(row.condition), float(row.day))
        for row in grouped.itertuples(index=False)
    ]
    return grouped[["condition", "day", "aligned_day", "species", "ddpcr_copy_number", "ddpcr_sd_or_ci"]].reset_index(
        drop=True
    )


def _load_t87_raw_anchor_copy_targets(raw_dir: str | Path | None) -> pd.DataFrame:
    if raw_dir is None:
        return _empty_t87_copy_targets()
    path = Path(raw_dir) / "ddpcr.csv"
    if not path.exists():
        return _empty_t87_copy_targets()

    raw = pd.read_csv(path)
    required = {"condition", "species", "ddpcr_copy_number"}
    if not required.issubset(raw.columns):
        return _empty_t87_copy_targets()

    rows = raw[raw["condition"].isin(cfg.T87_CONDITION_TREATMENTS) & raw["species"].isin(cfg.SPECIES)].copy()
    if rows.empty:
        return _empty_t87_copy_targets()

    rows["day"] = rows["condition"].map(_t87_copy_target_start_day).astype(float)
    rows["aligned_day"] = T87_EXPERIMENT_START_DAY
    if "ddpcr_sd_or_ci" not in rows.columns:
        rows["ddpcr_sd_or_ci"] = float("nan")

    grouped = (
        rows.groupby(["condition", "day", "aligned_day", "species"], as_index=False)
        .agg(ddpcr_copy_number=("ddpcr_copy_number", "median"), ddpcr_sd_or_ci=("ddpcr_sd_or_ci", "median"))
        .sort_values(["condition", "species", "day"])
    )
    return grouped[["condition", "day", "aligned_day", "species", "ddpcr_copy_number", "ddpcr_sd_or_ci"]].reset_index(
        drop=True
    )


def _load_t87_copy_targets(raw_dir: str | Path | None) -> pd.DataFrame:
    filtered = _load_t87_filtered_copy_targets(T87_FILTERED_DDPCR_SOURCE)
    anchors = _load_t87_raw_anchor_copy_targets(raw_dir)
    if filtered.empty:
        return anchors
    if anchors.empty:
        return filtered

    combined = pd.concat(
        [filtered.assign(_priority=0), anchors.assign(_priority=1)],
        ignore_index=True,
    )
    combined = combined.sort_values("_priority").drop_duplicates(["condition", "day", "species"], keep="first")
    return combined.drop(columns="_priority").sort_values(["condition", "species", "day"]).reset_index(drop=True)


def _complete_t87_treatment_groups(frames: dict[str, pd.DataFrame]) -> list[tuple[str, tuple[str, ...]]]:
    return [
        (group_title, group_conditions)
        for group_title, group_conditions in T87_TREATMENT_GROUPS
        if all(condition in frames for condition in group_conditions)
    ]


def _plot_t87_log10_state_counts(
    groups: list[tuple[str, tuple[str, ...]]],
    frames: dict[str, pd.DataFrame],
    metadata: dict[str, dict],
    week1_cell_counts: dict[str, float],
    save_path: str | Path,
) -> plt.Figure:
    fig, axes = plt.subplots(
        len(groups),
        4,
        figsize=(16, max(3.6, 3.4 * len(groups))),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row_idx, (group_title, group_conditions) in enumerate(groups):
        for col_idx, condition in enumerate(group_conditions):
            ax = axes[row_idx, col_idx]
            frame = frames[condition]
            times = frame["time"].to_numpy(dtype=float)
            scale = _condition_cell_count_scale(condition, frame, metadata.get(condition, {}), week1_cell_counts)
            for state_idx, state_name in enumerate(cfg.STATE_NAMES):
                column = f"dominant_count_{_safe_token(state_name)}"
                counts = frame[column].to_numpy(dtype=float) * scale
                ax.plot(
                    times,
                    np.log10(np.clip(counts, 1.0, None)),
                    color=STATE_COLORS[state_idx],
                    linewidth=2,
                    label=state_name,
                )
            ax.set_title(_condition_dose_label(condition))
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
            if row_idx == len(groups) - 1:
                ax.set_xlabel("Time")
            if col_idx == 0:
                ax.set_ylabel(f"{group_title}\nlog10(cells)")

    axes[0, -1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("T87 log10 state cell counts")
    fig.tight_layout()
    return _save(fig, save_path)


def _terminal_aligned_frame_indices(frame: pd.DataFrame, target_count: int) -> list[int]:
    if frame.empty:
        return []
    if len(frame) <= target_count:
        return list(range(len(frame)))

    times = frame["time"].to_numpy(dtype=float)
    target_times = np.linspace(float(times[0]), float(times[-1]), num=target_count)
    selected_positions: list[int] = []
    previous_position = -1

    for target_idx, target_time in enumerate(target_times):
        remaining_slots = target_count - target_idx - 1
        min_position = previous_position + 1
        max_position = len(frame) - remaining_slots - 1
        if target_idx == target_count - 1:
            position = len(frame) - 1
        else:
            candidate_times = times[min_position : max_position + 1]
            position = min_position + int(np.argmin(np.abs(candidate_times - target_time)))
        selected_positions.append(position)
        previous_position = position

    return selected_positions


def _initial_t87_copy_target(copy_targets: pd.DataFrame, condition: str, species_name: str) -> float:
    if copy_targets.empty:
        return float("nan")
    start_day = _t87_copy_target_start_day(condition)
    rows = copy_targets[
        (copy_targets["condition"] == condition)
        & (copy_targets["species"] == species_name)
        & np.isclose(copy_targets["day"].astype(float), start_day)
    ]
    if rows.empty:
        return float("nan")
    return float(rows["ddpcr_copy_number"].median())


def _t87_copy_plot_points(
    frame: pd.DataFrame,
    copy_targets: pd.DataFrame,
    condition: str,
    species_name: str,
) -> pd.DataFrame:
    column = f"mean_copy_{species_name}"
    points = frame.loc[
        (frame["time"].astype(float) >= 0.0)
        & (frame["time"].astype(float) <= float(cfg.T87_TREATMENT_END_TIME))
        & frame[column].notna(),
        ["time", column],
    ].rename(columns={column: "mean_copy"})

    initial_copy = _initial_t87_copy_target(copy_targets, condition, species_name)
    if np.isfinite(initial_copy):
        points = pd.concat(
            [pd.DataFrame([{"time": 0.0, "mean_copy": initial_copy}]), points],
            ignore_index=True,
        )

    if points.empty:
        return pd.DataFrame(columns=["time", "aligned_day", "mean_copy"])

    points = points.astype({"time": float, "mean_copy": float})
    points = points.groupby("time", as_index=False)["mean_copy"].mean().sort_values("time").reset_index(drop=True)
    selected = points.iloc[_terminal_aligned_frame_indices(points, T87_COPY_NUMBER_TIMEPOINT_COUNT)].copy()
    selected["aligned_day"] = selected["time"].map(_t87_sim_time_to_aligned_day)
    return selected


def _plot_t87_copy_targets(
    ax: plt.Axes,
    copy_targets: pd.DataFrame,
    condition: str,
    species_name: str,
) -> None:
    if copy_targets.empty:
        return
    rows = copy_targets[(copy_targets["condition"] == condition) & (copy_targets["species"] == species_name)].copy()
    if rows.empty:
        return
    rows = rows.sort_values("aligned_day")
    ax.plot(
        rows["aligned_day"].to_numpy(dtype=float),
        rows["ddpcr_copy_number"].to_numpy(dtype=float),
        color=T87_CONDITION_COLORS[condition],
        linestyle=(0, (5, 3)),
        linewidth=1.35,
        alpha=0.68,
        zorder=2,
        label="_nolegend_",
    )
    yerr_values = rows["ddpcr_sd_or_ci"].to_numpy(dtype=float)
    yerr = np.where(np.isfinite(yerr_values), yerr_values, 0.0)
    ax.errorbar(
        rows["aligned_day"].to_numpy(dtype=float),
        rows["ddpcr_copy_number"].to_numpy(dtype=float),
        yerr=yerr,
        color=T87_CONDITION_COLORS[condition],
        marker="s",
        markerfacecolor="white",
        markeredgewidth=1.2,
        linestyle="none",
        capsize=2.5,
        markersize=4,
        linewidth=1.0,
        alpha=0.72,
        zorder=2.2,
        label="_nolegend_",
    )


def _plot_t87_ecdna_copy_number_by_treatment(
    groups: list[tuple[str, tuple[str, ...]]],
    frames: dict[str, pd.DataFrame],
    copy_targets: pd.DataFrame,
    save_path: str | Path,
) -> plt.Figure:
    fig, axes = plt.subplots(
        len(groups),
        cfg.N_SPECIES,
        figsize=(14, max(3.8, 3.6 * len(groups))),
        sharex=True,
        squeeze=False,
    )

    for row_idx, (group_title, group_conditions) in enumerate(groups):
        for species_idx, species_name in enumerate(cfg.SPECIES):
            ax = axes[row_idx, species_idx]
            for condition in group_conditions:
                plot_points = _t87_copy_plot_points(frames[condition], copy_targets, condition, species_name)
                if plot_points.empty:
                    continue
                ax.plot(
                    plot_points["aligned_day"].to_numpy(dtype=float),
                    plot_points["mean_copy"].to_numpy(dtype=float),
                    color=T87_CONDITION_COLORS[condition],
                    marker="o",
                    markersize=4,
                    linewidth=2.6,
                    alpha=0.95,
                    zorder=3,
                    label=_condition_dose_label(condition),
                )
                _plot_t87_copy_targets(ax, copy_targets, condition, species_name)
            ax.set_title(species_name)
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
            ax.set_xlim(T87_EXPERIMENT_START_DAY - 1.0, T87_EXPERIMENT_END_DAY + 1.0)
            ax.set_xticks(np.linspace(T87_EXPERIMENT_START_DAY, T87_EXPERIMENT_END_DAY, T87_COPY_NUMBER_TIMEPOINT_COUNT))
            if row_idx == len(groups) - 1:
                ax.set_xlabel("Aligned experimental day")
            if species_idx == 0:
                ax.set_ylabel(f"{group_title}\nMean copy number")
        condition_handles, condition_labels = axes[row_idx, -1].get_legend_handles_labels()
        condition_legend = axes[row_idx, -1].legend(
            condition_handles,
            condition_labels,
            frameon=False,
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
        )
        axes[row_idx, -1].add_artist(condition_legend)
        style_handles = [
            plt.Line2D([0], [0], color="#111827", marker="o", linewidth=2.6, markersize=4, label="Simulation"),
            plt.Line2D(
                [0],
                [0],
                color="#111827",
                marker="s",
                markerfacecolor="white",
                linestyle=(0, (5, 3)),
                linewidth=1.35,
                markersize=4,
                label="ddPCR",
            ),
        ]
        axes[row_idx, -1].legend(
            handles=style_handles,
            frameon=False,
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 0.48),
        )

    fig.suptitle("T87 ecDNA copy-number trajectories")
    fig.tight_layout()
    return _save(fig, save_path)


def _pooled_state_group_copy_stats(row: pd.Series, state_names: tuple[str, ...], species_name: str) -> tuple[float, float]:
    total_count = 0.0
    weighted_sum = 0.0
    weighted_second_moment = 0.0

    for state_name in state_names:
        state_token = _safe_token(state_name)
        count = float(row.get(f"dominant_count_{state_token}", 0.0))
        if not np.isfinite(count) or count <= 0.0:
            continue
        mean = float(row.get(f"state_mean_copy_{state_token}_{species_name}", float("nan")))
        variance = float(row.get(f"state_var_copy_{state_token}_{species_name}", float("nan")))
        if not np.isfinite(mean) or not np.isfinite(variance):
            continue
        variance = max(0.0, variance)
        total_count += count
        weighted_sum += count * mean
        weighted_second_moment += count * (variance + mean * mean)

    if total_count <= 0.0:
        return float("nan"), float("nan")

    mean = weighted_sum / total_count
    variance = max(0.0, weighted_second_moment / total_count - mean * mean)
    return float(mean), float(np.sqrt(variance))


def _endpoint_row_closest_to_treatment_end(frame: pd.DataFrame) -> pd.Series:
    times = frame["time"].to_numpy(dtype=float)
    cfg.require(times.size > 0, "T87 endpoint plot requires at least one recorded time.")
    endpoint_idx = int(np.argmin(np.abs(times - float(cfg.T87_TREATMENT_END_TIME))))
    return frame.iloc[endpoint_idx]


def _plot_t87_state_group_ecdna_endpoint_points(
    groups: list[tuple[str, tuple[str, ...]]],
    frames: dict[str, pd.DataFrame],
    save_path: str | Path,
) -> plt.Figure:
    state_groups = (("OPC+NPC", ("OPC-like", "NPC-like")), ("AC+MES", ("AC-like", "MES-like")))
    x_positions = np.arange(len(state_groups), dtype=float)

    fig, axes = plt.subplots(
        len(groups),
        cfg.N_SPECIES,
        figsize=(14, max(3.8, 3.6 * len(groups))),
        sharex=True,
        squeeze=False,
    )

    for row_idx, (group_title, group_conditions) in enumerate(groups):
        offsets = np.linspace(-0.24, 0.24, num=len(group_conditions))
        for species_idx, species_name in enumerate(cfg.SPECIES):
            ax = axes[row_idx, species_idx]
            for condition_idx, condition in enumerate(group_conditions):
                row = _endpoint_row_closest_to_treatment_end(frames[condition])
                means: list[float] = []
                errors: list[float] = []
                for _label, state_names in state_groups:
                    mean, sd = _pooled_state_group_copy_stats(row, state_names, species_name)
                    means.append(mean)
                    errors.append(0.0 if not np.isfinite(sd) else sd)
                mean_values = np.asarray(means, dtype=float)
                error_values = np.asarray(errors, dtype=float)
                valid = np.isfinite(mean_values)
                if not np.any(valid):
                    continue
                ax.errorbar(
                    x_positions[valid] + offsets[condition_idx],
                    mean_values[valid],
                    yerr=error_values[valid],
                    color=T87_CONDITION_COLORS[condition],
                    marker="o",
                    linestyle="none",
                    capsize=4,
                    markersize=5,
                    linewidth=1.4,
                    label=_condition_dose_label(condition),
                )
            ax.set_title(species_name)
            ax.set_xticks(x_positions, labels=[label for label, _state_names in state_groups])
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
            if species_idx == 0:
                ax.set_ylabel(f"{group_title}\nMean copy number")
        axes[row_idx, -1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    fig.suptitle("T87 endpoint ecDNA copy number by state group")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_t87_treatment_comparison_suite(
    output_dir: str | Path,
    *,
    raw_dir: str | Path | None = None,
    conditions: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    output = Path(output_dir)
    requested_conditions = conditions or tuple(
        dict.fromkeys(condition for _group_title, group_conditions in T87_TREATMENT_GROUPS for condition in group_conditions)
    )
    frames, metadata = _load_t87_condition_frames(output, tuple(requested_conditions))
    groups = _complete_t87_treatment_groups(frames)
    if not groups:
        return {}

    plot_dir = output / "t87_comparison_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    week1_cell_counts = _load_t87_week1_cell_counts(raw_dir)
    copy_targets = _load_t87_copy_targets(raw_dir)
    plot_specs = (
        (
            "01_log10_state_counts_by_condition.png",
            lambda path: _plot_t87_log10_state_counts(groups, frames, metadata, week1_cell_counts, path),
        ),
        (
            "02_ecdna_copy_number_by_treatment.png",
            lambda path: _plot_t87_ecdna_copy_number_by_treatment(groups, frames, copy_targets, path),
        ),
        (
            "03_state_group_ecdna_endpoint_points.png",
            lambda path: _plot_t87_state_group_ecdna_endpoint_points(groups, frames, path),
        ),
    )

    written: dict[str, Path] = {}
    for file_name, plot_func in plot_specs:
        path = plot_dir / file_name
        plot_func(path)
        written[file_name] = path
    return written


def _latest_nonempty_snapshot(result: SimulationResult) -> tuple[float, list[dict]]:
    for time, snapshot in zip(reversed(result.times), reversed(result.cell_snapshots)):
        if snapshot:
            return float(time), snapshot
    raise ValueError("Lineage state plot requires at least one non-empty cell snapshot.")


def _terminal_cell_ids(snapshot: list[dict], n_terminal_cells: int | None = None) -> list[int]:
    if n_terminal_cells is not None:
        cfg.require(n_terminal_cells > 0, "n_terminal_cells must be strictly positive.")
    terminal_ids = sorted(int(cell["cell_id"]) for cell in snapshot)
    cfg.require(bool(terminal_ids), "Cannot sample lineage paths from an empty snapshot.")
    if n_terminal_cells is None or len(terminal_ids) <= n_terminal_cells:
        return terminal_ids

    sampled_indices: list[int] = []
    for raw_index in np.rint(np.linspace(0, len(terminal_ids) - 1, num=n_terminal_cells)).astype(int).tolist():
        if raw_index not in sampled_indices:
            sampled_indices.append(raw_index)
    sampled_indices.sort()
    return [terminal_ids[index] for index in sampled_indices]


def plot_lineage_state_paths(
    result: SimulationResult,
    n_terminal_cells: int | None = None,
    title: str = "Lineage state paths",
    save_path: str | Path | None = None,
) -> plt.Figure:
    cfg.require(bool(result.times), "Lineage state plot requires recorded simulation times.")
    latest_time, latest_snapshot = _latest_nonempty_snapshot(result)
    selected_terminal_ids = _terminal_cell_ids(latest_snapshot, n_terminal_cells)

    initial_snapshot = result.cell_snapshots[0] if result.cell_snapshots else []
    parent_by_cell: dict[int, int | None] = {int(cell["cell_id"]): None for cell in initial_snapshot}
    children_by_parent: dict[int, list[int]] = defaultdict(list)
    birth_times: dict[int, float] = {int(cell["cell_id"]): float(result.times[0]) for cell in initial_snapshot}
    division_times: dict[int, float] = {}
    death_times: dict[int, float] = {}

    for event_time, event_type, cell_id, details in result.events:
        event_time = float(event_time)
        parent_id = int(cell_id)
        if event_type == "division":
            division_times[parent_id] = event_time
            for daughter_key in ("daughter_one", "daughter_two"):
                daughter = details[daughter_key]
                daughter_id = int(daughter["cell_id"])
                cfg.require(
                    int(daughter["parent_id"]) == parent_id,
                    "Division event daughter must point back to the recorded parent cell.",
                )
                parent_by_cell[daughter_id] = parent_id
                birth_times[daughter_id] = event_time
                children_by_parent[parent_id].append(daughter_id)
        elif event_type == "death":
            death_times[parent_id] = event_time

    included_cells: set[int] = set()
    for terminal_id in selected_terminal_ids:
        current_id: int | None = terminal_id
        while current_id is not None and current_id not in included_cells:
            included_cells.add(current_id)
            current_id = parent_by_cell.get(current_id)

    cfg.require(bool(included_cells), "Lineage state plot requires at least one sampled lineage.")

    state_points: dict[int, dict[float, int]] = {cell_id: {} for cell_id in included_cells}
    for time, snapshot in zip(result.times, result.cell_snapshots):
        for cell in snapshot:
            cell_id = int(cell["cell_id"])
            if cell_id in included_cells:
                state_points[cell_id][float(time)] = _dominant_state_index(cell["soft_state"])

    for event_time, event_type, cell_id, details in result.events:
        event_time = float(event_time)
        cell_id = int(cell_id)
        if event_type == "division" and cell_id in included_cells:
            state_points[cell_id][event_time] = _dominant_state_index(details["state_pre"]["soft_state"])
            for daughter_key in ("daughter_one", "daughter_two"):
                daughter = details[daughter_key]
                daughter_id = int(daughter["cell_id"])
                if daughter_id in included_cells:
                    state_points[daughter_id][event_time] = _dominant_state_index(daughter["soft_state"])
        elif event_type == "death" and cell_id in included_cells:
            state_points[cell_id][event_time] = _dominant_state_index(details["state_pre"]["soft_state"])

    leaf_rank = {cell_id: idx for idx, cell_id in enumerate(selected_terminal_ids)}
    filtered_children = {
        cell_id: [child_id for child_id in children_by_parent.get(cell_id, []) if child_id in included_cells]
        for cell_id in included_cells
    }
    leftmost_rank_cache: dict[int, int] = {}

    def leftmost_rank(cell_id: int) -> int:
        cached = leftmost_rank_cache.get(cell_id)
        if cached is not None:
            return cached
        children = filtered_children.get(cell_id, [])
        if not children:
            cfg.require(cell_id in leaf_rank, f"Included leaf cell {cell_id} must be one of the sampled terminals.")
            leftmost_rank_cache[cell_id] = leaf_rank[cell_id]
            return leaf_rank[cell_id]
        rank = min(leftmost_rank(child_id) for child_id in children)
        leftmost_rank_cache[cell_id] = rank
        return rank

    y_positions: dict[int, float] = {}

    def assign_y(cell_id: int) -> float:
        if cell_id in y_positions:
            return y_positions[cell_id]
        children = filtered_children.get(cell_id, [])
        if not children:
            y_positions[cell_id] = float(leaf_rank[cell_id])
            return y_positions[cell_id]
        ordered_children = sorted(children, key=leftmost_rank)
        child_positions = [assign_y(child_id) for child_id in ordered_children]
        y_positions[cell_id] = float(np.mean(child_positions))
        return y_positions[cell_id]

    roots = [cell_id for cell_id in included_cells if parent_by_cell.get(cell_id) is None or parent_by_cell.get(cell_id) not in included_cells]
    for root in sorted(roots, key=leftmost_rank):
        assign_y(root)

    fig, ax = plt.subplots(figsize=(12, max(4.0, 0.8 * len(selected_terminal_ids) + 2.0)))
    state_cmap = dict(zip(range(cfg.N_STATES), STATE_COLORS))

    for cell_id in included_cells:
        points = state_points[cell_id]
        sorted_times = sorted(points.keys())
        y = y_positions[cell_id]
        birth_time = birth_times.get(cell_id, min(sorted_times))
        end_time = death_times.get(cell_id, division_times.get(cell_id, latest_time))

        if birth_time in points:
            start_state = points[birth_time]
        else:
            earlier_times = [time for time in sorted_times if time <= birth_time]
            start_state = points[max(earlier_times)] if earlier_times else points[sorted_times[0]]

        ax.hlines(y, birth_time, end_time, color=state_cmap[start_state], linewidth=2.5, alpha=0.45)

        for time in sorted_times:
            ax.scatter(time, y, s=26, color=state_cmap[points[time]], edgecolors="none", zorder=3)

        parent_id = parent_by_cell.get(cell_id)
        if parent_id is not None and parent_id in included_cells:
            ax.plot(
                [birth_time, birth_time],
                [y_positions[parent_id], y],
                color="#9ca3af",
                linewidth=1.1,
                alpha=0.8,
            )

        if cell_id in death_times:
            ax.scatter(death_times[cell_id], y, marker="x", s=42, color="#111827", linewidths=1.3, zorder=4)

    ax.set_yticks([y_positions[cell_id] for cell_id in selected_terminal_ids])
    ax.set_yticklabels([f"Cell {cell_id}" for cell_id in selected_terminal_ids])
    ax.set_xlabel("Time")
    ax.set_title(title)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=STATE_COLORS[idx], markersize=7, label=state_name)
        for idx, state_name in enumerate(cfg.STATE_NAMES)
    ]
    legend_handles.append(plt.Line2D([0], [0], marker="x", color="#111827", linestyle="none", markersize=7, label="Death"))
    ax.legend(handles=legend_handles, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.tight_layout()
    return _save(fig, save_path)
